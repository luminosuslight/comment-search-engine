#!/usr/bin/python3

import csv
import json
import pickle
import os
import time
import shutil
import argparse
import sys
import array
import cProfile
import pstats
from base64 import b64encode, b64decode
from bisect import bisect_left
from math import log
from multiprocessing import Pool

import numpy as np

from nltk.corpus import stopwords

from gensim.utils import tokenize
from gensim.parsing.porter import PorterStemmer

csv.field_size_limit(2147483647)


MAX_INDEX_STRING_LENGTH = 8

COMMENT_ID_FIELD = 0
TEXT_FIELD = 3
REPLY_TO_FIELD = 5

guardian = False
if guardian:
    # original Guardians data set: 0 article_id, 1 author_id, 2 comment_id, 3 text, 4 parent_cid, 5 timestamp, 6 upvotes
    COMMENT_ID_FIELD = 2
    TEXT_FIELD = 3
    REPLY_TO_FIELD = 4


def print_profile(func):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        my_return_val = func(*args, **kwargs)

        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        ps.sort_stats('cumulative')
        ps.print_stats()
        return my_return_val
    return inner


def int_to_bytes(x):
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')


def int_from_bytes(xbytes):
    return int.from_bytes(xbytes, 'big')


def int_to_base64(x):
    # this functions is a performance bottleneck -> all in one line
    return b64encode(x.to_bytes((x.bit_length() + 7) // 8, 'big')).decode().rstrip('=')


def int_from_base64(s):
    return int.from_bytes(b64decode(s + '=='), 'big')


# pre-calculated values for better performance:
pos_as_base64 = [int_to_base64(i) for i in range(400)]
base64_to_pos = dict((int_to_base64(i), i) for i in range(128))


stopwords = set(stopwords.words("english"))
stemmer_fn = PorterStemmer().stem


def get_tokens(text):
    tokens = tokenize(text, lowercase=True)

    # add position numbers, remove punctuation and stem words:
    stemmed_words = [(pos, stemmer_fn(word)) for (pos, word) in enumerate(tokens) if
                     word not in stopwords and word.isalpha() or word.endswith("*")]
    return stemmed_words


def process_comment(c):
    comment_id, comment = c
    # 'text' is 4th item in comment
    return comment_id, comment, get_tokens(comment[3])


def delta_compress_numbers(numbers):
    last_pos = 0
    for i in range(len(numbers)):
        pos = numbers[i]
        numbers[i] = pos - last_pos
        last_pos = pos


def delta_uncompress_numbers(numbers):
    last_pos = 0
    for i in range(len(numbers)):
        delta = numbers[i]
        pos = last_pos + delta
        numbers[i] = pos
        last_pos = pos


class SearchEngine(object):

    def __init__(self, data_filename):
        self._data_filename = data_filename
        self._seek_filename = 'index/seek.dat'
        self._postings_filename = 'index/postings.dat'
        self._index_part_dir = 'index_parts'
        self._stats_filename = 'index/stats.dat'
        self._comment_lengths_filename = 'index/lengths.dat'
        self._reply_to_filename = 'index/reply_to.dat'
        self._reply_seek_filename = 'index/reply_seek.dat'

        self._postings = {}
        self._seek_list = None
        self._seek_positions = None
        self._total_comment_length = 0
        self._comment_lengths_keys = array.array('L')  # unsigned long (64bit)
        self._comment_lengths_values = array.array('H')  # unsigned short (16bit)
        # self._comment_csv_ids = array.array('I')  # unsigned int (32bit)
        self._replies = {}

        self._reply_keys = array.array('L')
        self._reply_positions = array.array('L')
        self._reply_to_file = None
        self._postings_file = None
        self._data_file = None
        self._comment_count = 0
        self._avg_comment_length = 0
        self._reply_count = 0

        if os.path.exists(self._index_part_dir):
            shutil.rmtree(self._index_part_dir)
        os.makedirs(self._index_part_dir)
        if not os.path.exists('index'):
            os.makedirs('index')

    def create_index(self):
        if os.path.exists(self._postings_filename):
            print("Already indexed. Delete index to regenerate it.\n")
            return

        print("Indexing...")
        begin = time.time()
        self._index_data()
        duration = time.time() - begin
        print("Indexing completed (%.2fs)\n" % duration)

        print("Merging parts...")
        begin = time.time()
        self._merge()
        duration = time.time() - begin
        print("Merging parts completed (%.2fs)\n" % duration)

    def _index_data(self):
        self._total_comment_length = 0
        with open(self._data_filename, 'r', newline='') as csvfile:
            csvfile.readline()  # skip header
            # initial position:
            pos = csvfile.tell()
            comment_count = 0
            comment_chunk = []
            process_pool = Pool()

            try:
                for line in iter(csvfile.readline, ''):
                    comment_id = pos
                    comment = next(csv.reader([line]))
                    comment_chunk.append((comment_id, comment))
                    pos = csvfile.tell()
                    comment_count += 1

                    if len(comment_chunk) >= 50000:
                        comment_chunk = process_pool.map(process_comment, comment_chunk)
                        for comment_id, comment, tokens in comment_chunk:
                            self._index_comment(comment_id, comment, tokens)
                        del comment_chunk
                        comment_chunk = []
                        print("%d comments processed" % comment_count)

                    if not comment_count % 100000:
                        self._write_index_part_to_disk()

            except KeyboardInterrupt:
                print("Stopping indexing processes...")
                process_pool.terminate()
                process_pool.join()
                print("Indexing interrupted, continuing with merging...")
            except Exception as e:
                print(e)
                print("Something happened, trying to continue...")

            if comment_chunk:
                comment_chunk = map(process_comment, comment_chunk)
                for comment_id, comment, tokens in comment_chunk:
                    self._index_comment(comment_id, comment, tokens)
                del comment_chunk
                print("%d comments processed" % comment_count)
                self._write_index_part_to_disk()

        avg_comment_length = self._total_comment_length / comment_count
        print("Comment count:", comment_count)
        print("Average comment length:", avg_comment_length)
        print("Total token count:", self._total_comment_length)

        self._comment_count = comment_count
        self._avg_comment_length = avg_comment_length

        delta_compress_numbers(self._comment_lengths_keys)
        # delta_compress_numbers(self._comment_csv_ids)
        with open(self._comment_lengths_filename, 'w') as lengths_file:
            for i in range(len(self._comment_lengths_keys)):
                lengths_file.write("%d\n" % self._comment_lengths_keys[i])
                lengths_file.write("%d\n" % self._comment_lengths_values[i])
                # lengths_file.write("%d\n" % self._comment_csv_ids[i])
            del self._comment_lengths_keys  # free memory
            del self._comment_lengths_values
            # del self._comment_csv_ids
            self._comment_lengths_keys = array.array('L')
            self._comment_lengths_values = array.array('H')
            # self._comment_csv_ids = array.array('I')

        print("writing replies...")
        self._write_replies_to_disk()

    def _index_comment(self, comment_id, comment, tokens):
        self._total_comment_length += len(tokens)
        self._comment_lengths_keys.append(comment_id)
        self._comment_lengths_values.append(len(tokens))
        # self._comment_csv_ids.append(int(comment[COMMENT_ID_FIELD]))

        # append these occurrences of the tokens to the posting lists:
        for pos, word in tokens:
            self._postings.setdefault(word[:MAX_INDEX_STRING_LENGTH], []).append((comment_id, pos))

        if comment[REPLY_TO_FIELD]:
            try:
                self._replies.setdefault(int(comment[REPLY_TO_FIELD]), array.array('L')).append(comment_id)
            except:
                pass

    def _write_index_part_to_disk(self):
        if not self._postings: return
        part_num = len(os.listdir(self._index_part_dir))
        filename = (self._index_part_dir + '/part%d') % part_num

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for token, postings in sorted(self._postings.items()):
                writer.writerow([token, json.dumps(postings, separators=(',', ':'))])

        # clear postings in memory:
        self._postings = {}
        print("Wrote index part to disk.")

    def _merge(self):
        parts = os.listdir(self._index_part_dir)
        readers = {}
        part_files = []

        # open part files:
        for part in parts:
            path = self._index_part_dir + "/" + part
            f = open(path, 'r', newline='')
            part_files.append(f)
            reader = csv.reader(f)
            readers[reader] = next(reader)

        postings_file = open(self._postings_filename, 'w', newline='')
        data_type = 'U%d' % MAX_INDEX_STRING_LENGTH
        seek_list = np.ndarray([0], dtype=data_type)
        seek_positions = array.array('L')
        json_loads = json.loads

        # This is a K-Way merging algorithm:
        # In each step we look at the next word in each part file.
        # Then we sort them alphabetically and take the first word.
        # If more than one part file has this word as the next, we merge their lists
        # and set their cursor to the next value for the next round.
        # At last we write the merged list to the final postings file.
        while True:
            words = (item[0] for item in readers.values())
            word = min(words)
            if not word:
                break

            complete_posting = []
            for reader, item in readers.items():
                if item[0] == word:
                    complete_posting += json_loads(item[1])
                    readers[reader] = next(reader, ["", None])

            # store index of posting in postings file:
            seek_list.resize((seek_list.shape[0] + 1,))
            seek_list[-1] = word
            seek_positions.append(postings_file.tell())
            # write posting list to postings file:
            postings_file.write(self._postings_to_string(complete_posting) + '\n')

        # write seek list to disk:
        seek_list = self._compress_seek_list(seek_list)
        delta_compress_numbers(seek_positions)
        self._write_seek_file(seek_list, seek_positions)

        # close files:
        postings_file.close()
        for part_file in part_files:
            part_file.close()

        with open(self._stats_filename, 'wb') as stats_file:
            stats_file.write(pickle.dumps({"doc_count": self._comment_count,
                                           "avg_doc_length": self._avg_comment_length,
                                           "dict_size": len(seek_list),
                                           "reply_count": self._reply_count}))

    def _postings_to_string(self, postings):
        s = []
        for cid, pos in postings:
            s += (int_to_base64(cid), ":", (pos_as_base64[pos] if pos < len(pos_as_base64) else int_to_base64(pos)), ";")
        return ''.join(s)

    def _postings_from_string(self, s):
        postings = []
        for post in s.rstrip(";").split(";"):
            cid, pos = post.split(":")
            cid = int.from_bytes(b64decode(cid + '=='), 'big')  # inline int_from_base64(cid)
            pos = base64_to_pos.get(pos, None) or int.from_bytes(b64decode(pos + '=='), 'big')  # inline int_from_base64(pos)
            postings.append((cid, pos))
        return postings

    def _compress_seek_list(self, seek_list):
        compressed = []
        last_word = ""
        for word in seek_list:
            prefix_length = 0
            for i in range(min(9, len(word), len(last_word))):
                if word[i] != last_word[i]:
                    break
                prefix_length = i + 1
            compressed.append(str(prefix_length) + word[prefix_length:])
            last_word = word
        return compressed

    def _uncompress_seek_list(self, compressed):
        last_word = ""
        i = 0
        for word in compressed:
            prefix_length = int(word[0])
            real_word = last_word[:prefix_length] + word[1:]
            compressed[i] = real_word
            i += 1
            last_word = real_word
        return compressed

    def _write_seek_file(self, tokens, positions):
        if not len(tokens) == len(positions):
            print("Seek tokens and positions have different size. Aborting.")
            exit(-1)
        with open(self._seek_filename, 'w') as seek_file:
            for i in range(len(tokens)):
                seek_file.write(tokens[i] + "\n")
                seek_file.write(int_to_base64(positions[i]) + "\n")

    def _read_seek_file(self, dict_size):
        data_type = 'U%d' % MAX_INDEX_STRING_LENGTH
        tokens = np.ndarray([dict_size], dtype=data_type)
        positions = array.array('L', [0]) * dict_size
        with open(self._seek_filename, 'r') as seek_file:
            for i in range(dict_size):
                token = seek_file.readline()
                tokens[i] = token[:-1]
                positions[i] = int_from_base64(seek_file.readline()[:-1])
        return tokens, positions

    def _load_comment_lengths(self):
        comment_lengths_keys = array.array('L', [0]) * self._comment_count
        comment_lengths_values = array.array('H', [0]) * self._comment_count
        # comment_csv_ids = array.array('I', [0]) * self._comment_count
        with open(self._comment_lengths_filename, 'r') as lengths_file:
            readline_fn = lengths_file.readline
            for i in range(self._comment_count):
                comment_lengths_keys[i] = int(readline_fn()[:-1])
                comment_lengths_values[i] = int(readline_fn()[:-1])
                #comment_csv_ids[i] = int(readline_fn()[:-1])
        delta_uncompress_numbers(comment_lengths_keys)
        # delta_uncompress_numbers(comment_csv_ids)
        self._comment_lengths_keys = comment_lengths_keys
        self._comment_lengths_values = comment_lengths_values
        # self._comment_csv_ids = comment_csv_ids

    def _write_replies_to_disk(self):
        reply_keys = array.array('L', [0]) * len(self._replies)
        reply_positions = array.array('L', [0]) * len(self._replies)
        i = 0
        with open(self._reply_to_filename, 'w') as reply_to_file:
            for cid, replies in sorted(self._replies.items()):
                reply_keys[i] = cid
                reply_positions[i] = reply_to_file.tell()
                compressed_replies = (int_to_base64(r) for r in replies)
                reply_to_file.write(','.join(compressed_replies) + '\n')
                i += 1

        self._reply_count = len(self._replies)
        del self._replies
        self._replies = None

        delta_compress_numbers(reply_keys)
        delta_compress_numbers(reply_positions)
        with open(self._reply_seek_filename, 'w') as reply_seek_file:
            for i in range(len(reply_keys)):
                reply_seek_file.write("%d\n" % reply_keys[i])
                reply_seek_file.write("%d\n" % reply_positions[i])

    def _load_replies(self, reply_count):
        reply_keys = array.array('L', [0]) * reply_count
        reply_positions = array.array('L', [0]) * reply_count
        with open(self._reply_seek_filename, 'r') as reply_seek_file:
            for i in range(reply_count):
                reply_keys[i] = int(reply_seek_file.readline()[:-1])
                reply_positions[i] = int(reply_seek_file.readline()[:-1])
        delta_uncompress_numbers(reply_keys)
        delta_uncompress_numbers(reply_positions)
        self._reply_keys = reply_keys
        self._reply_positions = reply_positions

    def load_index(self):
        begin = time.time()

        with open(self._stats_filename, 'rb') as stats_file:
            stats = pickle.load(stats_file)
            self._comment_count = stats["doc_count"]
            self._avg_comment_length = stats["avg_doc_length"]
            dict_size = stats["dict_size"]
            self._reply_count = stats["reply_count"]

        compressed_seek_list, compressed_seek_positions = self._read_seek_file(dict_size)

        duration = time.time() - begin
        print("Loaded seek list in %.2fms." % (duration * 1000))
        begin = time.time()

        self._seek_list = self._uncompress_seek_list(compressed_seek_list)
        delta_uncompress_numbers(compressed_seek_positions)
        self._seek_positions = compressed_seek_positions

        duration = time.time() - begin
        print("Uncompressed seek list in %.2fms." % (duration * 1000))
        begin = time.time()

        self._load_comment_lengths()

        duration = time.time() - begin
        print("Loaded comment lengths in %.2fms." % (duration * 1000))
        begin = time.time()

        self._load_replies(self._reply_count)

        self._postings_file = open(self._postings_filename, 'r', newline='')
        self._data_file = open(self._data_filename, 'r', newline='')
        self._reply_to_file = open(self._reply_to_filename, 'r', newline='')

        duration = time.time() - begin
        print("Loaded reply index in %.2fms." % (duration * 1000))
        print("")

    def search(self, query, top_k):
        query = query.replace('"', "'").replace('`', "'").replace('’', "'").replace("”", "'")
        if query.startswith("ReplyTo:"):
            if any(word in query for word in (" AND ", " OR ", " NOT ")):
                print("Not supported.")
                return []
            return self.get_replies(int(query[len("ReplyTo:"):].split()[0]), top_k)
        elif any(word in query for word in (" AND ", " OR ", " NOT ")):
            if "'" in query or "*" in query:
                print("Not supported.")
                return []
            return self._search_binary(query, top_k)
        else:
            return self._search_BM25(query, top_k)

    def _search_binary(self, query, top_k):
        print("Searching using binary model: ", query)
        begin = time.time()

        is_phrase_query = query.startswith("'") and query.endswith("'")

        is_phrase_query_joker= query.startswith("'") and query.endswith("*")

        tokens = get_tokens(query)

        # find all postings:
        all_postings = []
        for pos, word in tokens:
            all_postings.append(set([comment_id for comment_id, pos in self._get_postings(word)]))

        # combine postings according to operator:
        if " NOT " in query:
            if len(all_postings) != 2:
                print("Invalid query")
                return []
            relevant_doc_ids = all_postings[0] - all_postings[1]
        elif " OR " in query:
            relevant_doc_ids = set.union(*all_postings)
        else:  # AND
            relevant_doc_ids = set.intersection(*all_postings)

        duration = time.time() - begin
        print("Found %d results in %.2fms." % (len(relevant_doc_ids), duration * 1000))
        begin = time.time()

        # materialize results:
        results = []
        for doc_id in relevant_doc_ids:
            comment = self.get_comment(doc_id)
            comment.append(1.0)  # score
            results.append(comment)
            if top_k and len(results) >= top_k:
                break

        duration = time.time() - begin
        print("Materialized %s results in %.2fms." % (top_k or "all", duration * 1000))
        begin = time.time()

        if is_phrase_query:
            simple_query = query.replace("'", "").lower()
            for comment in results.copy():
                if simple_query not in comment[3].lower():
                    results.remove(comment)
            duration = time.time() - begin
            print("Phrase query matched %d results in %.2fms." % (len(results), duration * 1000))

        elif is_phrase_query_joker:
            simple_query = query.replace("*", "").lower()
            simple_query = simple_query.replace("'", "").lower()
            for comment in results.copy():
                if simple_query not in comment[3].lower():
                    results.remove(comment)
            duration = time.time() - begin
            print("Phrase query matched %d results in %.2fms." % (len(results), duration * 1000))

        return results

    def get_doc_length(self, comment_id):
        i = bisect_left(self._comment_lengths_keys, comment_id)
        if i == len(self._comment_lengths_keys):
            print("Could not find length for comment", comment_id)
            return self._avg_comment_length
        return self._comment_lengths_values[i]

    def _search_BM25(self, query, top_k):
        print("Searching using BM25: ", query)
        begin = time.time()

        is_phrase_query = query.startswith("'") and query.endswith("'")
        is_phrase_query_joker= query.startswith("'") and query.endswith("'*")

        # if is_phrase_query and query.endswith("*"):
        #     print("Not supported.")
        #     return []
        tokens = get_tokens(query)
        query_results = {}
        avg_doc_length = self._avg_comment_length

        for pos, word in tokens:
            qf = len([w for pos, w in tokens if w == word])
            postings = self._get_postings(word)
            postings_count = len(postings)
            comment_ids = set([comment_id for comment_id, pos in postings])
            num_docs_containing_word = len(comment_ids)
            i = 0
            while i < postings_count:
                comment_id, pos = postings[i]
                term_count_in_doc = 0
                while i < postings_count and postings[i][0] == comment_id:
                    term_count_in_doc += 1
                    i += 1
                doc_length = self.get_doc_length(comment_id)
                score = self._bm25_score(num_docs_containing_word, term_count_in_doc, qf, doc_length, avg_doc_length)
                if comment_id in query_results:
                    query_results[comment_id] += score
                else:
                    query_results[comment_id] = score

        duration = time.time() - begin
        print("Found %d results in %.2fms." % (len(query_results), duration * 1000))
        begin = time.time()

        # materialize results:
        results = []
        simple_query = query.replace("'", "").lower()
        if is_phrase_query_joker:
            simple_query = simple_query.replace("*", "").lower()

        materialize_count = 0
        for comment_id, score in sorted(query_results.items(), key=lambda x: x[1], reverse=True):
            comment = self.get_comment(comment_id)
            materialize_count += 1
            if is_phrase_query and simple_query not in comment[3].lower():
                continue
            if is_phrase_query_joker and simple_query not in comment[3].lower():
                continue
            comment.append(score)
            results.append(comment)
            if len(results) >= top_k:
                break

        duration = time.time() - begin
        print("Materialized %d results to find %s results in %.2fms." %
              (materialize_count, "top %d" % top_k if top_k else "all", duration * 1000))
        if not results:
            print("No result found.")

        return results

    def _get_postings(self, word):
        # (it is ok to cut off * suffix if word is longer than INDEX_STRING_LENGTH,
        # because all those words end up in the same posting list anyway)
        word = word[:MAX_INDEX_STRING_LENGTH]
        is_prefix = word.endswith("*")
        if is_prefix:
            word = word[:-1]

        i = bisect_left(self._seek_list, word)
        postings = []
        if is_prefix:
            if not i or i == len(self._seek_list):
                print("Word is not in seek list:", word)
                return []

            while True:
                token = self._seek_list[i]
                if not token.startswith(word):
                    break
                pos = self._seek_positions[i]
                self._postings_file.seek(pos)
                line = self._postings_file.readline()
                postings = postings + json.loads(line)
                i = i+1

        else:  # exact match
            if i == len(self._seek_list) or self._seek_list[i] != word:
                print("Word is not in seek list:", word)
                return []

            pos = self._seek_positions[i]
            self._postings_file.seek(pos)
            line = self._postings_file.readline()
            postings = self._postings_from_string(line[:-1])
        return postings

    def _bm25_score(self, n, f, qf, ld, l_avg):
        # n = docs containing term
        # f = count of term in doc
        # qf = count of term in query
        # ld = length of doc
        # l_avg = avg length of docs

        k1 = 1.2
        k3 = 100.0
        b = 0.75
        r = 0.0
        R = 0.0
        N = self._comment_count
        K = k1 * ((1 - b) + b * (float(ld) / float(l_avg)))

        first = log(((r + 0.5) / (R - r + 0.5)) / ((n - r + 0.5) / (N - n - R + r + 0.5)))
        second = ((k1 + 1.0) * f) / (K + f)
        third = ((k3 + 1.0) * qf) / (k3 + qf)
        return first * second * third

    def get_comment(self, comment_id):
        # comment_id is byte offset in the data file
        self._data_file.seek(comment_id)
        line = self._data_file.readline()
        comment = next(csv.reader([line]))
        return comment

    def get_replies(self, csv_id, top_k):
        i = bisect_left(self._reply_keys, csv_id)
        if i == len(self._reply_keys) or self._reply_keys[i] != csv_id:
            return []
        pos = self._reply_positions[i]
        self._reply_to_file.seek(pos)
        line = self._reply_to_file.readline()
        replies = [int_from_base64(r) for r in line.split(',')]
        return (self.get_comment(cid) for cid in replies[:top_k or len(replies)])

    def search_and_write_to_file(self, query, top_k, out_filename, print_ids_only):
        print("\nOut File: %s, TopN: %s, Query: %s" % (out_filename, top_k, query))
        result = self.search(query, top_k)
        with open(out_filename, 'w') as out_file:
            for comment in result:
                if print_ids_only:
                    out_file.write("%s\n" % comment[COMMENT_ID_FIELD])
                else:
                    out_file.write("%s, %s\n" % (comment[COMMENT_ID_FIELD], comment[TEXT_FIELD]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="a txt file with one boolean, keyword, phrase, ReplyTo, or Index query per line")
    parser.add_argument("--topN", help="the maximum number of search hits to be printed", type=int)
    parser.add_argument("--printIdsOnly", help="print only commentIds and not ids and their corresponding comments",
                        action="store_true")
    args = parser.parse_args()

    searchEngine = SearchEngine('comments.csv')

    if args.query.startswith('Index:'):
        # custom name for CSV is not supported, because it is expected at './comments.csv' as
        # the data source while executing the queries
        searchEngine.create_index()
    else:
        with open(args.query, 'r',encoding="utf-8") as file:
            queries = [q[:-1] for q in file.readlines()]

        searchEngine.load_index()

        for i, query in enumerate(queries):
            searchEngine.search_and_write_to_file(query, args.topN, "query%d.txt" % (i+1,), args.printIdsOnly)

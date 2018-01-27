import csv
import json
import pickle
import os
import time
import shutil
from base64 import b64encode, b64decode
import cProfile
import pstats
import sys
import array
import memory_profiler
import numpy as np
from bisect import bisect_left
from math import log
from multiprocessing import Pool

from nltk.corpus import stopwords

from gensim.utils import tokenize
from gensim.parsing.porter import PorterStemmer

csv.field_size_limit(2147483647)


INDEX_STRING_LENGTH = 8


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
    b = b64decode(s + '==')
    x = int_from_bytes(b)
    return x


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
        self._comment_lengths_keys = array.array('i')
        self._comment_lengths_values = array.array('H')  # unsigned short
        self._replies = {}

        self._reply_keys = array.array('i')
        self._reply_positions = array.array('i')
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

                    #if not comment_count % 200000:
                    #    break

            except KeyboardInterrupt:
                print("Stopping indexing processes...")
                process_pool.terminate()
                process_pool.join()
                print("Indexing interrupted, continuing with merging...")

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
        with open(self._comment_lengths_filename, 'w') as lengths_file:
            for i in range(len(self._comment_lengths_keys)):
                lengths_file.write("%d\n" % self._comment_lengths_keys[i])
                lengths_file.write("%d\n" % self._comment_lengths_values[i])
            del self._comment_lengths_keys  # free memory
            del self._comment_lengths_values
            self._comment_lengths_keys = array.array('i')  # signed int
            self._comment_lengths_values = array.array('H')  # unsigned short

        print("writing replies...")
        self._write_replies_to_disk()

    def _index_comment(self, comment_id, comment, tokens):
        self._total_comment_length += len(tokens)
        self._comment_lengths_keys.append(comment_id)
        self._comment_lengths_values.append(len(tokens))

        # append these occurrences of the tokens to the posting lists:
        for pos, word in tokens:
            self._postings.setdefault(word[:INDEX_STRING_LENGTH], []).append((comment_id, pos))

        reply_to_field = 5  # as specified in the mail
        # reply_to_field = 4  # as in the guardian data set
        if comment[reply_to_field]:
            self._replies.setdefault(int(comment[reply_to_field]), array.array('i')).append(comment_id)

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
        data_type = 'U%d' % INDEX_STRING_LENGTH
        seek_list = np.ndarray([0], dtype=data_type)
        seek_positions = array.array('i')
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
        # this function is a performance bottleneck -> use append instead of +
        s = []
        for cid, pos in postings:
            s += (int_to_base64(cid), ":", (pos_as_base64[pos] if pos < len(pos_as_base64) else int_to_base64(pos)), ";")
        return ''.join(s)

    def _postings_from_string(self, s):
        postings = []
        for post in s.rstrip(";").split(";"):
            cid, pos = post.split(":")
            cid = int_from_base64(cid)
            pos = base64_to_pos.get(pos, None) or int_from_base64(pos)
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
        data_type = 'U%d' % INDEX_STRING_LENGTH
        tokens = np.ndarray([dict_size], dtype=data_type)
        positions = array.array('l', [0]) * dict_size
        with open(self._seek_filename, 'r') as seek_file:
            for i in range(dict_size):
                token = seek_file.readline()
                tokens[i] = token[:-1]
                positions[i] = int_from_base64(seek_file.readline()[:-1])
        return tokens, positions

    def _load_comment_lengths(self):
        comment_lengths_keys = array.array('i', [0]) * self._comment_count
        comment_lengths_values = array.array('H', [0]) * self._comment_count  # unsigned short
        with open(self._comment_lengths_filename, 'r') as lengths_file:
            readline_fn = lengths_file.readline
            for i in range(self._comment_count):
                comment_lengths_keys[i] = int(readline_fn()[:-1])
                comment_lengths_values[i] = int(readline_fn()[:-1])
        delta_uncompress_numbers(comment_lengths_keys)
        self._comment_lengths_keys = comment_lengths_keys
        self._comment_lengths_values = comment_lengths_values

    def _write_replies_to_disk(self):
        reply_keys = array.array('i', [0]) * len(self._replies)
        reply_positions = array.array('i', [0]) * len(self._replies)
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
        reply_keys = array.array('i', [0]) * reply_count
        reply_positions = array.array('i', [0]) * reply_count
        with open(self._reply_seek_filename, 'r') as reply_seek_file:
            for i in range(reply_count):
                reply_keys[i] = int(reply_seek_file.readline()[:-1])
                reply_positions[i] = int(reply_seek_file.readline()[:-1])
        delta_uncompress_numbers(reply_keys)
        delta_uncompress_numbers(reply_positions)
        self._reply_keys = reply_keys
        self._reply_positions = reply_positions

    #@memory_profiler.profile
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
        if query.startswith("ReplyTo:"):
            self.print_replies(int(query[len("ReplyTo:"):]), top_k)
        elif any(word in query for word in (" AND ", " OR ", " NOT ")):
            return self._search_binary(query, top_k)
        else:
            return self._search_BM25(query, top_k)

    def _search_binary(self, query, top_k):
        print("Searching using binary model: ", query)
        begin = time.time()

        is_phrase_query = query.startswith("'") and query.endswith("'")
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
            if len(results) >= top_k:
                break

        duration = time.time() - begin
        print("Materialized %d results in %.2fms." % (top_k, duration * 1000))
        begin = time.time()

        if is_phrase_query:
            simple_query = query.replace("'", "").lower()
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
        materialize_count = 0
        for comment_id, score in sorted(query_results.items(), key=lambda x: x[1], reverse=True):
            comment = self.get_comment(comment_id)
            materialize_count += 1
            if is_phrase_query and simple_query not in comment[3].lower():
                continue
            comment.append(score)
            results.append(comment)
            if len(results) >= top_k:
                break

        duration = time.time() - begin
        print("Materialized %d results to find top %d in %.2fms." % (materialize_count, top_k, duration * 1000))
        if not results:
            print("No result found.")

        return results

    def _get_postings(self, word):
        # (it is ok to cut off * suffix if word is longer than INDEX_STRING_LENGTH,
        # because all those words end up in the same posting list anyway)
        word = word[:INDEX_STRING_LENGTH]
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

    def get_comment_by_line_number(self, cid):
        # Finds a comment in a csv file
        # The id has to be the line number and the file has to be ordered
        # Binary search would need up to log(file_size) seeks, while
        # this approach needs only ~log(n)/4 seeks plus a sequential read
        # of up to 50 lines which is much more disk friendly and therefore faster
        self._data_file.seek(0, os.SEEK_END)
        file_size = self._data_file.tell()
        avg_comment_size = file_size / self._comment_count

        estimated_position = cid * avg_comment_size
        self._data_file.seek(estimated_position)
        self._data_file.readline()  # discard partial line
        line = self._data_file.readline()
        comment = next(csv.reader([line]))
        nr = int(comment[0])
        diff = cid - nr
        if diff == 0:
            return comment

        while diff < 0 or diff > 50:
            estimated_position = max(0, min(estimated_position + (diff * avg_comment_size), file_size - 1))
            self._data_file.seek(estimated_position)
            self._data_file.readline()  # discard partial line
            line = self._data_file.readline()
            comment = next(csv.reader([line]))
            nr = int(comment[0])
            diff = cid - nr
            if diff == 0:
                return comment

        for i in range(51):
            line = self._data_file.readline()
            comment = next(csv.reader([line]))
            nr = int(comment[0])
            diff = cid - nr
            if diff == 0:
                return comment

        print("Could not find comment with id:", cid)
        return None

    def _get_replies(self, cid):
        i = bisect_left(self._reply_keys, cid)
        if i == len(self._reply_keys) or self._reply_keys[i] != cid:
            return []
        pos = self._reply_positions[i]
        self._reply_to_file.seek(pos)
        line = self._reply_to_file.readline()
        replies = [int_from_base64(r) for r in line.split(',')]
        return replies

    def print_replies(self, cid, top_k):
        begin = time.time()
        comment = self.get_comment_by_line_number(cid)
        print("\nSearching for replies to the following comment:\n%s" % comment[3])
        reply_ids = self._get_replies(cid)
        duration = time.time() - begin
        print("Found %d results in %.2fms." % (len(reply_ids), duration * 1000))
        begin = time.time()
        for rid in reply_ids[:top_k]:
            reply = self.get_comment(rid)
            print(" -> " + reply[3])
        duration = time.time() - begin
        print("Materialized %d results in %.2fms." % (min(len(reply_ids), top_k), duration * 1000))

    def print_assignment2_query_results(self):
        # print first 5 results for every query:
        # self.print_results("October", 5)
        self.print_results("jobs", 5)
        # self.print_results("Trump", 5)
        # self.print_results("hate", 5)
        self.print_results("party AND chancellor", 5)
        # self.print_results("party NOT politics", 1)
        # self.print_results("war OR conflict", 1)
        # self.print_results("euro* NOT europe", 1)
        self.print_results("publi* NOT moderation", 1)
        # self.print_results("'the european union'", 1)
        # self.print_results("'christmas market'", 1)
        # Assignment 4:
        self.print_results("christmas market", 5)
        #self.print_results("catalonia independence", 5)
        self.print_results("'european union'", 5)
        self.print_results("negotiate", 5)
        self.search("ReplyTo:300744", 5)
        #self.search("ReplyTo:300748", 5)
        #self.search("ReplyTo:26252", 5)
        #self.search("ReplyTo:157515", 5)

    def test(self):
        self.print_results("jobs", 5)
        #self.print_results("party AND chancellor", 5)
        #self.print_results("publi* NOT moderation", 1)
        #self.print_results("christmas market", 5)
        self.print_results("'european union'", 5)
        self.print_results("negotiate", 5)
        #self.search("ReplyTo:300744", 5)

    def print_results(self, query, top_k):
        print("\n".join(["%.1f - %s" % (c[-1], c[3]) for c in self.search(query, top_k)]) + "\n")


if __name__ == '__main__':
    searchEngine = SearchEngine('data/comments_new.csv')
    searchEngine.create_index()
    searchEngine.load_index()
    #searchEngine.print_assignment2_query_results()
    searchEngine.test()

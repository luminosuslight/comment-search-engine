import csv
import json
import pickle
import os
import time
import shutil
from bisect import bisect_left
from math import log

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

csv.field_size_limit(2147483647)


class SearchEngine(object):

    def __init__(self, data_filename):
        self._data_filename = data_filename
        self._seek_filename = 'seek.dat'
        self._postings_filename = 'postings.dat'
        self._index_part_dir = 'index_parts'

        self._postings = {}
        self._seek_list = None
        self._seek_positions = None

        self._postings_file = None
        self._data_file = None
        self._comment_count = 0
        self._avg_comment_length = 0

        self._stopwords = set(stopwords.words("english"))
        self._stemmer = PorterStemmer()

        if os.path.exists(self._index_part_dir):
            shutil.rmtree(self._index_part_dir)
        os.makedirs(self._index_part_dir)

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
        with open(self._data_filename, 'r', newline='') as csvfile:
            # initial position:
            pos = csvfile.tell()
            comment_count = 0

            for line in iter(csvfile.readline, ''):
                comment_id = pos
                comment = next(csv.reader([line]))
                self._index_comment(comment, comment_id)
                pos = csvfile.tell()

                comment_count += 1
                if not comment_count % 10000:
                    print("%d comments processed" % comment_count)
                if not comment_count % 50000:
                    self._write_index_part_to_disk()

            self._write_index_part_to_disk()
        # TODO: write comment_count to disk
        # TODO: save avg comment length

    def _index_comment(self, comment, comment_id):
        tokens = self.get_tokens(comment[3])  # 'text' is 4th item in comment

        # append these occurrences of the tokens to the posting lists:
        for pos, word in tokens:
            self._postings.setdefault(word, []).append((comment_id, pos))

    def get_tokens(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [word.lower() for word in tokens]

        # add position numbers, remove punctuation and stem words:
        stops = self._stopwords  # faster than multiple accesses to instance variable
        stemmed_words = [(pos, self._stemmer.stem(word)) for (pos, word) in enumerate(tokens) if
                         word not in stops and word.isalpha() or word.endswith("*")]
        return stemmed_words

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
        seek_list = []
        seek_positions = []

        # This is a K-Way merging algorithm:
        # In each step we look at the next word in each part file.
        # Then we sort them alphabetically and take the first word.
        # If more than one part file has this word as the next, we merge their lists
        # and set their cursor to the next value for the next round.
        # At last we write the merged list to the final postings file.
        while True:
            words = [item[0] for item in readers.values()]
            word = sorted(words)[0]
            if not word:
                break

            complete_posting = []
            for reader, item in readers.items():
                if item[0] == word:
                    complete_posting = complete_posting + (json.loads(item[1]))
                    readers[reader] = next(reader, ["", None])

            # store index of posting in postings file:
            seek_list.append(word)
            seek_positions.append(postings_file.tell())
            # write posting list to postings file:
            postings_file.write(json.dumps(complete_posting, separators=(',', ':')) + '\n')

        # write seek list to disk:
        with open(self._seek_filename, 'wb') as seek_file:
            seek_file.write(pickle.dumps((seek_list, seek_positions)))

        # close files:
        postings_file.close()
        for part_file in part_files:
            part_file.close()

    def load_index(self):
        with open(self._seek_filename, 'rb') as seek_file:
            self._seek_list, self._seek_positions = pickle.load(seek_file)

        self._postings_file = open(self._postings_filename, 'r', newline='')
        self._data_file = open(self._data_filename, 'r', newline='')
        self._comment_count = 680000  # TODO: replace with real value from disk
        self._avg_comment_length = 10  # TODO: replace with real value from disk

    def search(self, query):
        print("Searching: ", query)
        begin = time.time()

        is_phrase_query = query.startswith("'") and query.endswith("'")
        tokens = self.get_tokens(query)

        # find all postings:
        all_postings = []
        for pos, word in tokens:
            all_postings.append(set([comment_id for comment_id, pos in self.get_postings(word)]))

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
            results.append(comment)

        duration = time.time() - begin
        print("Materialized %d results in %.2fms." % (len(relevant_doc_ids), duration * 1000))
        begin = time.time()

        if is_phrase_query:
            simple_query = query.replace("'", "").lower()
            for comment in results.copy():
                if simple_query not in comment[3].lower():
                    results.remove(comment)
            duration = time.time() - begin
            print("Phrase query matched %d results in %.2fms." % (len(results), duration * 1000))

        return results

    def get_postings(self, word):
        is_prefix = word.endswith("*")
        word = word.replace("*", "")

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
            postings = json.loads(line)
        return postings

    def search_BM25(self, query, top_x):
        tokens = self.get_tokens(query)
        query_results = {}
        avg_comment_length = self._avg_comment_length

        for pos, word in tokens:
            qf = len([w for pos, w in tokens if w == word])
            postings = self.get_postings(word)
            comment_ids = set([comment_id for comment_id, pos in postings])
            num_docs_containing_word = len(comment_ids)
            for comment_id, pos in postings:
                raw_tf_in_comment = len([cid for cid, pos in postings if cid == comment_id])  # TODO: read tf from extra index file
                comment_length = 10  # TODO: read comment length from extra index file
                score = self._bm25_rank(num_docs_containing_word, raw_tf_in_comment, qf, comment_length, avg_comment_length)
                if comment_id in query_results:
                    query_results[comment_id] += score
                else:
                    query_results[comment_id] = score

        # materialize results:
        results = []
        for comment_id, score in sorted(query_results.items(), key=lambda x: x[1], reverse=True)[:top_x]:
            comment = self.get_comment(comment_id)
            results.append(comment)

        return results


    def _bm25_rank(self, n, f, qf, ld, l_avg):
        # n = docs containing term
        # f = count of term in doc
        # qf = count of term in query
        # ld = length of doc
        # l_avg = avg length of docs

        k1 = 1.2
        k3 = 100
        b = 0.75
        r = 0
        R = 0.0
        N = self._comment_count
        K = k1 * ((1 - b) + b * (float(ld) / float(l_avg)))

        first = log(((r + 0.5) / (R - r + 0.5)) / ((n - r + 0.5) / (N - n - R + r + 0.5)))
        second = ((k1 + 1) * f) / (K + f)
        third = ((k3 + 1) * qf) / (k3 + qf)
        return first * second * third

    def get_comment(self, comment_id):
        # comment_id is byte offset in the data file
        self._data_file.seek(comment_id)
        line = self._data_file.readline()
        comment = next(csv.reader([line]))
        return comment

    def print_assignment2_query_results(self):
        # print first 5 results for every query:
        self.print_results("October", 5)
        self.print_results("jobs", 5)
        self.print_results("Trump", 5)
        self.print_results("hate", 5)
        self.print_results("party AND chancellor", 1)
        self.print_results("party NOT politics", 1)
        self.print_results("war OR conflict", 1)
        self.print_results("euro* NOT europe", 1)
        self.print_results("publi* NOT moderation", 1)
        self.print_results("'the european union'", 1)
        self.print_results("'christmas market'", 1)

        print("\n".join([c[3] for c in self.search_BM25("angela merkel", 5)]) + "\n")

    def print_results(self, query, count):
        print("\n".join([c[3] for c in self.search(query)[:count]]) + "\n")


if __name__ == '__main__':
    searchEngine = SearchEngine('data/comments_new.csv')
    searchEngine.create_index()
    searchEngine.load_index()
    searchEngine.print_assignment2_query_results()

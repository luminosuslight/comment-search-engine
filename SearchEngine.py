import csv
import json
import pickle
import os
import time
import shutil

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class SearchEngine(object):

    def __init__(self, data_filename):
        self._data_filename = data_filename
        self._seek_filename = 'seek.dat'
        self._postings_filename = 'postings.dat'
        self._index_part_dir = 'index_parts'

        self._postings = {}
        self._seek_list = None
        self._postings_file = None
        self._data_file = None

        self._stopwords = set(stopwords.words("english"))
        self._stemmer = PorterStemmer()

        if os.path.exists(self._index_part_dir):
            shutil.rmtree(self._index_part_dir)
        os.makedirs(self._index_part_dir)

    def index(self):
        if os.path.exists(self._postings_filename):
            print("Already indexed.")
            return

        print("Indexing...")
        begin = time.time()

        with open(self._data_filename, 'r', newline='') as csvfile:
            # skip header:
            csvfile.readline()
            # initial position:
            pos = csvfile.tell()
            comment_count = 0
            for line in iter(csvfile.readline, ''):
                comment_id = pos
                comment = next(csv.reader([line]))
                self._index_comment(comment, comment_id)
                comment_count += 1
                if not comment_count % 10000:
                    print("%d comments processed" % comment_count)
                    self._write_index_part_to_disk()
                pos = csvfile.tell()

        duration = time.time() - begin
        print("Indexing completed (%.2fs)" % duration)

        begin = time.time()
        self._merge()
        duration = time.time() - begin
        print("Merging parts completed (%.2fs)" % duration)

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
                         word not in stops and word.isalpha()]
        return stemmed_words

    def _write_index_part_to_disk(self):
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
        print("Merging parts...")
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
        seek_list = {}

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
            seek_list[word] = postings_file.tell()
            # write posting list to postings file:
            postings_file.write(json.dumps(complete_posting, separators=(',', ':')) + '\n')

        # write seek list to disk:
        with open(self._seek_filename, 'wb') as seek_file:
            seek_file.write(pickle.dumps(seek_list))

        # close files:
        postings_file.close()
        for part_file in part_files:
            part_file.close()

    def load_index(self):
        with open(self._seek_filename, 'rb') as seek_file:
            self._seek_list = pickle.load(seek_file)

        self._postings_file = open(self._postings_filename, 'r', newline='')
        self._data_file = open(self._data_filename, 'r', newline='')

    def get_postings(self, word):
        pos = self._seek_list.get(word, -1)
        if pos < 0:
            print("Word is not in seek list:", word)
            return []
        self._postings_file.seek(pos)
        line = self._postings_file.readline()
        return json.loads(line)

    def search(self, query):
        print("Searching: ", query)
        begin = time.time()

        tokens = self.get_tokens(query)

        all_postings = []
        for pos, word in tokens:
            all_postings.append(set([doc_id for doc_id, pos in self.get_postings(word)]))

        relevant_doc_ids = set.intersection(*all_postings)

        results = []
        for doc_id in relevant_doc_ids:
            comment = self.get_comment(doc_id)
            results.append(comment)

        duration = time.time() - begin
        print("Found %d results in %.2fms." % (len(results), duration * 1000))
        return results

    def get_comment(self, comment_id):
        # comment_id is byte offset in the data file
        self._data_file.seek(comment_id)
        line = self._data_file.readline()
        comment = next(csv.reader([line]))
        return comment

    def print_assignment2_query_results(self):
        # print first 5 results for every query:
        print("\n".join([c[3] for c in self.search("October")[:5]]))
        print("\n".join([c[3] for c in self.search("jobs")[:5]]))
        print("\n".join([c[3] for c in self.search("Trump")[:5]]))
        print("\n".join([c[3] for c in self.search("hate")[:5]]))
        print("\n".join([c[3] for c in self.search("some cool stuff")[:5]]))


if __name__ == '__main__':
    searchEngine = SearchEngine('data/comments_2017.csv')

    searchEngine.index()

    searchEngine.load_index()

    searchEngine.print_assignment2_query_results()

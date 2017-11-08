import csv

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class SearchEngine(object):

    def __init__(self):
        self._postings = {}
        self._comment_count = 0
        self._com_id_to_short_id = {}
        self._short_id_to_comment = {}

        self._stopwords = set(stopwords.words("english"))
        self._stemmer = PorterStemmer()

    def index(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            i = 0  # stop after 10 comments for debugging
            for row in reader:
                self.index_comment(row)
                i += 1
                if i > 10:
                    break

    def index_comment(self, comment):
        # get short id (the 'id' in comment is a rather long string):
        com_id = comment['id']
        if com_id in self._com_id_to_short_id:
            # comment is already in index
            return
        else:
            short_id = self._comment_count
            self._comment_count += 1
            self._com_id_to_short_id[com_id] = short_id
            self._short_id_to_comment[short_id] = comment

        # tokenize comment text:
        tokens = self.get_tokens(comment['text'])

        # write tokens to index:
        for pos, word in tokens:
            self._postings.setdefault(word, []).append((short_id, pos))

    def get_tokens(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [word.lower() for word in tokens]

        # add position numbers, remove punctuation and stem words:
        stops = self._stopwords  # faster than multiple accesses to instance variable
        stemmed_words = [(pos, self._stemmer.stem(word)) for (pos, word) in enumerate(tokens) if word not in stops and word.isalpha()]
        return stemmed_words

    def load_index(self, directory):
        # the index is stored completely in memory at the moment
        # -> no need to load an index from disk
        pass

    def search(self, query):
        print("Searching: ", query)
        results = []
        tokens = self.get_tokens(query)

        all_postings = []
        for pos, word in tokens:
            all_postings.append(set([doc_id for doc_id, pos in self._postings.get(word, [])]))

        relevant_doc_ids = set.intersection(*all_postings)

        for doc_id in relevant_doc_ids:
            comment = self._short_id_to_comment[doc_id]
            results.append(comment)

        return results

    def print_assignment2_query_results(self):
        # print first 3 results of every search result:
        print("\n".join([c['text'] for c in searchEngine.search("October")[:3]]))
        print("\n".join([c['text'] for c in searchEngine.search("jobs")[:3]]))
        print("\n".join([c['text'] for c in searchEngine.search("Trump")[:3]]))
        print("\n".join([c['text'] for c in searchEngine.search("hate")[:3]]))
        print("\n".join([c['text'] for c in searchEngine.search("art")[:3]]))


if __name__ == '__main__':
    searchEngine = SearchEngine()

    searchEngine.index('data/comments.csv')

    searchEngine.print_assignment2_query_results()

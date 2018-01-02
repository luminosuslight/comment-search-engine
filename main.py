from search_engine import SearchEngine

if __name__ == '__main__':
    searchEngine = SearchEngine('data/comments_new.csv')
    searchEngine.create_index()
    searchEngine.load_index()
    searchEngine.print_assignment2_query_results()
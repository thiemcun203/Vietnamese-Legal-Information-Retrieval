from code.bm25.bm25_class import BM25
from code.preprocess.preprocess_functions import bm25_preprocess
import numpy as np
import pickle
  
class BM25Utilities:
    def __init__(self, bm25_text, k1 = 1.2, b = 0.65, look_up = False, bm25_corpus = None, limit = 10):
        print("BM25Utilities initializing...")
        self.bm25 = BM25(k1=k1, b=b)
        self.bm25.fit(bm25_text)

        self.look_up = look_up
        if look_up == True:  
            self.corpus_look_up = bm25_corpus['law_article_id']  
    
        self.limit = limit
        
    def get_bm25_ranking(self, query, preprocessed = True):
        if not preprocessed:
            query = bm25_preprocess(query)
        query = query.split()
        scores = self.bm25.search(query=query)
        top_ids = np.argsort(scores)
        top_ids = top_ids[::-1]
        return top_ids

    def get_top_k_relevance(self, top_ids):

        assert self.look_up == True 
        top_k_results = [self.corpus_look_up[i] for i in top_ids][:self.limit]

        return top_k_results
    
def get_bm25_text(): 
    inp = open("midpoints/bm25_text", "rb")
    text = pickle.load(inp)
    return text
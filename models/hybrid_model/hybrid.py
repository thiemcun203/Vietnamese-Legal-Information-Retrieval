import sys, os
sys.path.append(os.getcwd())
from models.bm25 import *
from models.biencoder_model import *
class Hybrid:
    def __init__(self, biencoder, bm25, topk = 10, const = 20, biencoder_rate = 1.4, bm25_rate = 1.5):
        self.biencoder = biencoder
        self.bm25 = bm25
        self.raw_topk = 100
        self.topk = topk
        self.const = const
        self.biencoder_rate = biencoder_rate
        self.bm25_rate = bm25_rate
    
    def query(self, segmented_question, limit = 10):
        bm25_results = self.bm25.query(segmented_question, limit = self.raw_topk)
        biencoder_results = self.biencoder.query(segmented_question, topk = self.raw_topk)
        
        bm25_results_dict = {k:i+1 for i, k in enumerate(bm25_results.keys())}
        biencoder_results_dict = {k:i+1 for i, k in enumerate(biencoder_results.keys())}

        topk = self.topk
        const = self.const
        biencoder_rate = self.biencoder_rate
        bm25_rate = self.bm25_rate
        rrf_results = {}

        for docs in biencoder_results_dict.keys():
            score = 0
            score += biencoder_rate/(const + biencoder_results_dict[docs])
            if docs in bm25_results_dict:
                score += bm25_rate/(const+ bm25_results_dict[docs])
            else:
                score += bm25_rate/(const + topk)
                # score += 0
            rrf_results[docs] = score
        # {doc_id: score}
        
        rrf_lst = [k for k, v in sorted(rrf_results.items(), key=lambda item: item[1], reverse= True)]
        content_results = {}
        for doc in rrf_lst[:self.topk]:
            content_results[doc] = bm25_results.get(doc, []) + biencoder_results.get(doc, [])
            
        return content_results
    
if __name__ == "__main__":
    biencoder = BiEncoder()
    bm25 = BM25()
    bm25.fit()
    hybrid = Hybrid(biencoder, bm25)
    print(hybrid.query('khái_niệm quỹ đại_chúng'))
    
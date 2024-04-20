import numpy as np

def calculate_sub_score(ranking, k):
    index = np.arange(ranking.shape[0])
    score = np.vstack((index, ranking)) 
    score = score.T[score[1].argsort()].T
    score = score[0] + k 
    score = np.power(score*1.0, -1) 
    return score

def calculate_rrf_ranking(bm25_ranking, encoder_ranking, k = 60):
    index = np.arange(bm25_ranking.shape[0])
    bm25_score = calculate_sub_score(bm25_ranking, k)
    encoder_score = calculate_sub_score(encoder_ranking, k)

    rrf_score = bm25_score + encoder_score
    rrf_ranking = np.vstack((rrf_score, index))
    rrf_ranking = rrf_ranking.T[rrf_ranking[0].argsort()].T
    rrf_ranking = rrf_ranking[1]
    return rrf_ranking

class RRFUtilities:
    def __init__(self, bm25_utilities, encoder_utilities, device, look_up = True, rrf_corpus = None, limit = 10):
        print("RRFUtilities initializing...")
        self.bm25_utilities = bm25_utilities
        self.encoder_utilities = encoder_utilities
        self.device = device

        self.look_up = look_up
        if look_up == True:  
            self.corpus_look_up = rrf_corpus['law_article_id']  
    
        self.limit = limit
  
    def get_rrf_ranking(self, bm25_query, encoded_content, preprocessed = True):
        bm25_ranking = self.bm25_utilities.get_bm25_ranking(bm25_query, preprocessed=preprocessed)
        encoder_ranking = self.encoder_utilities.get_encoder_ranking(encoded_content, preprocessed=preprocessed)
        rrf_ranking = calculate_rrf_ranking(bm25_ranking=bm25_ranking, encoder_ranking=encoder_ranking, k=60)
        return rrf_ranking
    
    def get_top_k_relevance(self, top_ids):

        assert self.look_up == True 
        top_k_results = [self.corpus_look_up[i] for i in top_ids][:self.limit]

        return top_k_results

 
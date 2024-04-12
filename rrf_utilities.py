 
import numpy as np 
from tqdm.auto import tqdm
tqdm.pandas() 

def rrf(bm25_ranking, cosine_similarity_ranking, k=60):
    rrf_score = dict()
    for idx, doc_idx in enumerate(bm25_ranking):
        if doc_idx not in rrf_score:
            rrf_score[doc_idx] = 0
        rrf_score[doc_idx] += 1/(k + idx)

    for idx, doc_idx in enumerate(cosine_similarity_ranking):
        rrf_score[doc_idx] += 1/(k + idx)

    new_ranking = [i[0] for i in sorted(list(rrf_score.items()), key = lambda x: x[1], reverse=True)]
    return np.array(new_ranking)
 
def get_top_relevance_ids_in_batches(bm25_query, cos_sim_query, ranking, bm25_ranking = None, cos_sim_ranking = None, unique = True): 
    if ranking == "rrf":  
        top_ids = rrf(bm25_ranking, cos_sim_ranking) 
 
    return top_ids

limit = 10
def get_top_k_relevance(top_ids, corpus_look_up, unique, k = limit):
    if unique == True:
        multi_top_k_results = [corpus_look_up[i] for i in top_ids]
        top_k_results = list()
        for result in multi_top_k_results: 
            if len(top_k_results) == k:
                break
            else:
                if result not in top_k_results:
                    top_k_results.append(result)
    else:
        top_k_results = [corpus_look_up[i] for i in top_ids][:k]

    return top_k_results
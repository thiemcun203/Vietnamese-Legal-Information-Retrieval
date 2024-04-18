from code.bm25.bm25_utilities import BM25Utilities, get_bm25_text
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

def bm25_get_top_k_relevance(query, bm25_utilities):
    top_ids = bm25_utilities.get_bm25_ranking(query, preprocessed = True)
    top_k_relevance = bm25_utilities.get_top_k_relevance(top_ids = top_ids)
    return top_k_relevance

def bm25_process():
    bm25_text = get_bm25_text()
    bm25_corpus = pd.read_hdf("midpoints\\bm25_corpus.h5")
    bm25_utilities = BM25Utilities(bm25_text=bm25_text, k1 = 1.2, b = 0.65, look_up=True, bm25_corpus=bm25_corpus, limit = 10)

    bm25_test_qna = pd.read_hdf("midpoints\\bm25_test_qna.h5")
    print("Getting top BM25 relevances for qna dataset...")
    bm25_test_qna['top_relevance'] = bm25_test_qna.progress_apply(lambda x: bm25_get_top_k_relevance(x.segmented_question, bm25_utilities=bm25_utilities), axis = 1)
    bm25_test_qna.to_csv("output\\bm25_test_qna_output.csv")
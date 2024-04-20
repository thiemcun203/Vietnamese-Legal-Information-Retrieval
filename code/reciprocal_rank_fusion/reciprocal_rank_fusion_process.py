from code.encoder.encoder_utilities import EncoderUtilities
from code.bm25.bm25_utilities import BM25Utilities, get_bm25_text
from code.reciprocal_rank_fusion.reciprocal_rank_fusion_utilities import RRFUtilities
from tqdm import tqdm
import pandas as pd
import torch
tqdm.pandas()

def rrf_get_top_k_relevance(bm25_query, encoded_content, rrf_utilities):
    top_ids = rrf_utilities.get_rrf_ranking(bm25_query, encoded_content, preprocessed = True)
    top_k_relevance = rrf_utilities.get_top_k_relevance(top_ids = top_ids)
    return top_k_relevance

def rrf_process(): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    encoder_corpus = pd.read_hdf("midpoints\encoder_corpus.h5")
    encoder_utilities = EncoderUtilities(device=device, encoder_corpus=encoder_corpus, look_up=False)

    bm25_text = get_bm25_text() 
    bm25_utilities = BM25Utilities(bm25_text=bm25_text, k1 = 1.2, b = 0.65, look_up=False)

    rrf_corpus = pd.read_hdf("midpoints\\rrf_corpus.h5")
    rrf_utilities = RRFUtilities(bm25_utilities=bm25_utilities, encoder_utilities=encoder_utilities, device=device, look_up=True, 
                                 rrf_corpus=rrf_corpus, limit = 10)

    rrf_test_qna = pd.read_hdf("midpoints\\rrf_test_qna.h5")
    print("Getting top RRF relevances for qna dataset...")
    rrf_test_qna['top_relevance'] = rrf_test_qna.progress_apply(lambda x: rrf_get_top_k_relevance(x.segmented_question, x.encoded_question, rrf_utilities=rrf_utilities), axis = 1)
    rrf_test_qna = rrf_test_qna.drop(columns = ["encoded_question"])
    rrf_test_qna.to_csv("output\\rrf_test_qna_output.csv")
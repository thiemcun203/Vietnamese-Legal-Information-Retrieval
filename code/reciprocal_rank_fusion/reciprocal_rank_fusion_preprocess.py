import os.path
from code.bm25.bm25_preprocess import preprocess_for_bm25
from code.encoder.encoder_preprocess import preprocess_for_encoder
import pandas as pd

def preprocess_for_reciprocal_rank_fusion(batch_size = 32):
    if not os.path.isfile("midpoints\\bm25_corpus.h5"):
        print("RRF needs BM25 preprocessed data.")
        preprocess_for_bm25()
    if not os.path.isfile("midpoints\\encoder_corpus.h5"):
        print("RRF needs Encoder preprocessed data.")
        preprocess_for_encoder(batch_size=batch_size)
    
    print("Making RRF corpus...")
    rrf_corpus = pd.read_hdf("midpoints\\bm25_corpus.h5")
    encoder_corpus = pd.read_hdf("midpoints\\encoder_corpus.h5")
    rrf_corpus['encoded_content'] = encoder_corpus['encoded_content'].copy()

    del encoder_corpus

    rrf_corpus.to_hdf("midpoints\\rrf_corpus.h5", key = "rrf_corpus")

    print("Making RRF qna...")
    rrf_test_qna = pd.read_hdf("midpoints\\bm25_test_qna.h5")
    encoder_test_qna = pd.read_hdf("midpoints\\encoder_test_qna.h5")
    rrf_test_qna['encoded_question'] = encoder_test_qna['encoded_question']

    del encoder_test_qna

    rrf_test_qna.to_hdf("midpoints\\rrf_test_qna.h5", key = "rrf_test_qna")
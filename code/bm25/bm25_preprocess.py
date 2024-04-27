import pandas as pd
from code.preprocess.preprocess_functions import * 
from tqdm import tqdm
tqdm.pandas()
import pickle

def preprocess_for_bm25():
    print("Preprocessing corpus for BM25...")
    path_to_corpus = "input/rdrsegmenter_legal_corpus.csv"
    corpus = pd.read_csv(path_to_corpus) 
    corpus['segmented_title_content'] = corpus.progress_apply(lambda x: bm25_preprocess(x.segmented_title_content, skip_word_segment=True), axis = 1)
    corpus.to_hdf("midpoints/bm25_corpus.h5", key="bm25_corpus")

    print("Converting corpus...")
    documents = [document for document in corpus['segmented_title_content']]
    texts = list()
    for documents in tqdm(documents):
        texts.append([word for word in documents.lower().split() if word not in list_stopwords])
    out = open("midpoints/bm25_text", "wb")
    pickle.dump(texts, out)

    print("Preprocessing qna for BM25...")
    path_to_qna = "input/rdrsegmenter_testqna.csv"
    qna = pd.read_csv(path_to_qna)
    qna['segmented_question'] = qna.progress_apply(lambda x: bm25_preprocess(x.segmented_question), axis = 1)
    qna.to_hdf("midpoints/bm25_test_qna.h5", key="bm25_qna")
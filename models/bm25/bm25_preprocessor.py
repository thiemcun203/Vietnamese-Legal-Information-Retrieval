import os, pickle
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from ..utils import bm25_preprocess, list_stopwords, logger

class BM25Preprocessor:
    def __init__(self):
        self.root_dir = os.getcwd()
        self.processed_data_dir = os.getcwd() + "/models/bm25/processed_data"
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)
            logger.info("Created directory: models/bm25/processed_data")

    def preprocess_corpus(self, raw_corpus_path:str, skip_word_segment:bool=True):
        """
        Preprocess the corpus for BM25 ranking.
        :param raw_corpus_path: str, path to csv file containing the raw corpus
        :return: None
        """

        os.chdir(self.root_dir)
        corpus_path = os.getcwd() + raw_corpus_path
        os.chdir(self.processed_data_dir)

        # -------------------- Get bm25_full_corpus.h5: all corpus tabular information --------------------
        if os.path.exists("bm25_full_corpus.h5"):
            logger.info("bm25_full_corpus.h5 already exists.")
        else:
            corpus = pd.read_csv(corpus_path)
            corpus['segmented_title_content'] = corpus.progress_apply(lambda x: bm25_preprocess(x.segmented_title_content, skip_word_segment=skip_word_segment), axis = 1)
            corpus.to_hdf("bm25_full_corpus.h5", key="bm25_corpus")
            logger.info("Saved bm25_full_corpus.h5")

        # -------------------- Get bm25_only_text: only text --------------------
        if os.path.exists("bm25_only_text"):
            logger.info("bm25_only_text already exists.")
        else:
            documents = [document for document in corpus.segmented_title_content]
            texts = list()
            for documents in tqdm(documents):
                texts.append([word for word in documents.lower().split() if word not in list_stopwords])
            pickle.dump(texts, open("bm25_only_text", "wb"))
            logger.info("Saved bm25_only_text")
    
    def preprocess_qna(self, raw_qna_path:str, skip_word_segment:bool=True):
        """
        Preprocess the QnA for BM25 ranking.
        :param raw_qna_path: str, path to csv file containing the raw QnA
        :return: None
        """

        os.chdir(self.root_dir)
        qna_path = os.getcwd() + raw_qna_path
        os.chdir(self.processed_data_dir)

        # -------------------- Get bm25_test_qna.h5: all qna tabular information --------------------
        if os.path.exists("bm25_test_qna.h5"):
            logger.info("bm25_test_qna.h5 already exists.")
        else:
            qna = pd.read_csv(qna_path)
            qna['segmented_question'] = qna.progress_apply(lambda x: bm25_preprocess(x.segmented_question, skip_word_segment=skip_word_segment), axis = 1)
            qna.to_hdf("bm25_test_qna.h5", key="bm25_qna")
            logger.info("Saved bm25_test_qna.h5")
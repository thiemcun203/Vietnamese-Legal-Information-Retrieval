import os
import json
import math
import pickle
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict

from ..utils import bm25_preprocess, RetrievalMetrics, logger

class BM25:
    def __init__(
            self, 
            path_to_full_corpus:str="models/bm25/processed_data/bm25_full_corpus.h5", 
            path_to_only_text:str="models/bm25/processed_data/bm25_only_text",
            k1=1.2,
            b=0.65
        ):

        self.k1 = k1
        self.b = b
        
        self.idf = {}
        
        self.split_lookup = self._create_split_lookup(path_to_full_corpus)
        logger.info("Split lookup created.")

        self.corpus = pickle.load(open(path_to_only_text, "rb"))
        self.doc_count = 0
        self.doc_lens = []
        self.avgdl = 0

        self.inverted_index = {}
        logger.info("BM25 model initialized with k1={} and b={}".format(k1, b))

        self.root_dir = os.getcwd()
        self.prefitted_bm25_dir = os.getcwd() + "/models/bm25/prefitted_bm25"
        if not os.path.exists(self.prefitted_bm25_dir):
            os.makedirs(self.prefitted_bm25_dir)
            logger.info("Created directory: models/bm25/prefitted_bm25")
    
    def fit(self):
        """
        Fit the various statistics that are needed to calculate BM25 ranking for documents in a corpus. Save the fitted model.
        :return: None
        """

        # -------------------- Obtain general corpus information --------------------
        self.doc_count = len(self.corpus)
        self.doc_lens = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_lens) / self.doc_count

        # -------------------- Obtain inverted index --------------------
        for idx, doc in enumerate(self.corpus):
            doc = dict(Counter(doc))
            for word, count in doc.items():
                if word not in self.inverted_index:
                    self.inverted_index[word] = defaultdict(int)
                self.inverted_index[word][idx] = count
        logger.info("Inverted index created.")

        # -------------------- Create idf dictionary --------------------
        self.idf = self._create_idf_dict()
        logger.info("IDF dictionary created.")

        # -------------------- Save fitted BM25 model --------------------
        os.chdir(self.root_dir)
        pickle.dump(self, open(self.prefitted_bm25_dir + "/prefitted_bm25_model", "wb"))
        logger.info("Fitted BM25 model saved.")

    def search(self, query:str, preprocess_query=True):
        """
        Search the corpus for documents that are relevant to the query.
        :param query: 
        :param preprocess_query: bool: whether to preprocess the query. For example, in .test(), set this to False.
        :return: list of floats
        """
        if preprocess_query:
            query = bm25_preprocess(query, skip_word_segment=False)
        scores = [0] * self.doc_count
        for word in query.split():
            if word not in self.inverted_index:
                continue
            doc_list = self.inverted_index[word]
            for doc_id in doc_list.keys():
                scores[doc_id] += self._bm25(doc_id, word)
        return scores

    def _bm25(self, doc_id, word):
        """
        Calculate the BM25 score for a document and a query term.
        :param doc_id: int
        :param word: string
        :return: float
        """
        tf = self.inverted_index[word][doc_id]
        idf = self.idf[word]
        doc_len = self.doc_lens[doc_id]
        numerator = idf * tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
        return numerator / denominator
    
    def _create_idf_dict(self):
        """
        Create a dictionary of inverse document frequencies.
        :return: dict
        """
        idf = {}
        for word in self.inverted_index.keys():
            idf[word] = math.log((self.doc_count - len(self.inverted_index[word]) + 0.5) / (len(self.inverted_index[word]) + 0.5) + 1)
        return idf
    
    def _create_split_lookup(self, path_to_full_corpus):
        """
        Create a dictionary that maps the index of a split in the corpus to the ID of the full document as well as the actual segmented content.
        :return: dict
        """
        split_lookup = {}
        full_corpus = pd.read_hdf(path_to_full_corpus)
        for idx, row in tqdm(full_corpus.iterrows(), total=len(full_corpus)):
            split_lookup[idx] = {
                "split_id": idx,
                "law_article_id": row.law_article_id,
                "segmented_title_content": row.segmented_title_content
            }
        return split_lookup
    
    def infer(self, query:str, limit:int=10, preprocess_query=True, log_results=True):
        """
        Infer the top k most relevant documents to a query.
        :param query: string
        :param limit: int
        :param preprocess_query: bool: whether to preprocess the query. For example, in .test(), set this to False.
        :param log_results: bool: whether to log the results to the console. For example, in .test(), set this to False.
        :return: list of dicts
        """
        scores = self.search(query, preprocess_query=preprocess_query)
        top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        top_k_relevance = []
        for idx in top_ids:
            top_k_relevance.append(self.split_lookup[idx])
        
        if log_results:
            logger.info("Inferred top {} documents for query: {}".format(limit, query))
        for i, doc in enumerate(top_k_relevance):
            if log_results:
                logger.info("Rank {}: {}".format(i+1, doc))
        return top_k_relevance

    def test(self, path_to_test_qna:str="models/bm25/processed_data/bm25_test_qna.h5", limit:int=10):
        """
        Test the BM25 model on a QnA dataset.
        :param path_to_test_qna: str
        :param limit: int
        :return: None
        """
        results = {}
        test_qna = pd.read_hdf(path_to_test_qna)
        for _, row in tqdm(test_qna.iterrows(), total=len(test_qna)):
            question_id = row.question_id
            question = row.segmented_question
            # Take 10 times the limit to ensure that there are enough distinct articles
            results[question_id] = self.infer(question, 20*limit, preprocess_query=False, log_results=False)

        # -------------------- Save retrieval results and calculate metrics --------------------
        os.chdir(self.root_dir)
        if not os.path.exists("models/bm25/output"):
            os.makedirs("models/bm25/output")
            logger.info("Created directory: models/bm25/output")
        
        full_retrieval_results_path = f"models/bm25/output/{limit}_bm25_test_results_full.json"
        truncated_retrieval_results_path = f"models/bm25/output/{limit}_bm25_test_results_truncated.json"
        distinct_retrieval_results_path = f"models/bm25/output/{limit}_bm25_test_results_distinct.json"

        true_results_path = "data/true_results/true_test_results.json"

        # ---------- Save full retrieval results ----------
        with open(full_retrieval_results_path, "w") as f:
            json.dump(results, f, indent=4)
        
        # ---------- Truncate retrieval results ----------
        truncated_results = {}
        for question_id, docs in results.items():
            truncated_results[question_id] = [doc["law_article_id"] for doc in docs]
        with open(truncated_retrieval_results_path, "w") as f:
            json.dump(truncated_results, f, indent=4)
        
        # ---------- Distinct-article retrieval results ----------
        distinct_results = {}
        for question_id, docs in results.items():
            distinct_results[question_id] = []
            for doc in docs:
                if doc["law_article_id"] not in distinct_results[question_id]:
                    distinct_results[question_id].append(doc["law_article_id"])
                if len(distinct_results[question_id]) == limit:
                    break
        with open(distinct_retrieval_results_path, "w") as f:
            json.dump(distinct_results, f, indent=4)

        # Calculate metrics
        metrics = RetrievalMetrics(
            retrieval_results=distinct_retrieval_results_path,
            true_results=true_results_path
        )
        
        precision = metrics.get_precision_at_k(k=limit)
        recall = metrics.get_recall_at_k(k=limit)
        f2 = metrics.get_f_beta_score_at_k(k=limit, beta=2)
        mrr = metrics.get_mrr()
        map = metrics.get_map_at_k(k=limit)
        logger.info("Precision@{}: {:.2f}".format(limit, precision))
        logger.info("Recall@{}: {:.2f}".format(limit, recall))
        logger.info("F2@{}: {:.2f}".format(limit, f2))
        logger.info("MRR: {:.2f}".format(mrr))
        logger.info("MAP@{}: {:.2f}".format(limit, map))

        return {
            "precision": precision,
            "recall": recall,
            "f2": f2,
            "mrr": mrr,
            "map": map
        }
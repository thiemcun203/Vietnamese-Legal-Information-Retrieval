import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from ..bm25 import BM25, BM25Preprocessor
from ..utils import logger, RetrievalMetrics

dotenv_path = os.getcwd() + '/.env'
load_dotenv(dotenv_path)

client = QdrantClient(
    url=os.environ["SELF_HOST_URL"],
    api_key=os.environ["SELF_HOST_APIKEY"],
    timeout=60
)

class Reranker:
    def __init__(self, full_results_path:str=None, limit:int=10):
        """
        Initialize the Reranker model.
        :param full_results_path: str: path to the full BM25 results file. Example: "models/bm25/output/10_bm25_test_results_full.json"
        :param limit: int: the number of top splits to retrieve from BM25
        The class offers two ways to retrieve BM25 results. Either pass the full results file, or set the limit to the number of splits to retrieve.
        Only one of the two parameters should be passed.
        """
        
        logger.info("Initializing Reranker model...")
        self.root_dir = os.getcwd()
        # -------------------- Load BM25 model --------------------
        prefitted_bm25_path = os.getcwd() + "/models/bm25/prefitted_bm25/prefitted_bm25_model"
        if os.path.exists(prefitted_bm25_path):
            self.bm25 = pickle.load(open(prefitted_bm25_path, "rb"))
            logger.info(f"Prefitted BM25 model loaded from {prefitted_bm25_path}")
        else:
            preprocessor = BM25Preprocessor()
            preprocessor.preprocess_corpus("/input/rdrsegmenter_legal_corpus.csv", skip_word_segment=True)
            preprocessor.preprocess_qna("/input/rdrsegmenter_testqna.csv", skip_word_segment=True)
            self.bm25 = BM25()
            self.bm25.fit()
            logger.info("BM25 model fitted.")

        # -------------------- Load BM25 full results --------------------
        if os.path.exists(full_results_path):
            self.full_results = json.load(open(full_results_path))
            self.bm25_limit = len(self.full_results[list(self.full_results.keys())[0]]) // 20
            logger.info(f"BM25 full results loaded from {full_results_path}")
            logger.info("The BM25 limit is set to: {}".format(self.bm25_limit))
        else:
            self.bm25_limit = limit
            self.bm25.test(limit=self.bm25_limit)
            self.full_results = json.load(open(f"models/bm25/output/{self.bm25_limit}_bm25_test_results_full.json"))
            logger.info("BM25 full results computed.")
            logger.info("The BM25 limit is set to: {}".format(self.bm25_limit))

    def infer(self, query:str, limit:int=10, preprocess_query=True, log_results=True):
        """
        Infer the top k most relevant splits to a query, BASED ON reranking, via bi-encoder, the BM25 results.
        :param query: string
        :param limit: int
        :param preprocess_query: bool: whether to preprocess the query. For example, in .test(), set this to False.
        :param log_results: bool: whether to log the results to the console. For example, in .test(), set this to False.
        :return: list of dicts
        """

        model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder').to("cuda")
        # -------------------- Infer top k splits with BM25 --------------------
        top_k_relevance = self.bm25.infer(query, limit, preprocess_query=preprocess_query, log_results=log_results)
        split_ids = [split["split_id"] for split in top_k_relevance]
        query_embedding = model.encode(query)
        retrieved_results = client.retrieve(collection_name='embedding_legal_1', ids=split_ids, with_payload=False, with_vectors=True)
        split_embeddings = [result.vector for result in retrieved_results]

        # -------------------- Compute similarity with query --------------------
        query_embedding = np.array(query_embedding).reshape(1, -1)
        split_embeddings = np.array(split_embeddings)
        similarities = cosine_similarity(query_embedding, split_embeddings).flatten()
        top_ids = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        
        # -------------------- Sort top k splits by similarity --------------------
        top_k_relevance = [top_k_relevance[i] for i in top_ids]
        if log_results:
            logger.info("Reranker: Inferred top {} documents for query: {}".format(limit, query))
            for i, doc in enumerate(top_k_relevance):
                logger.info("Rank {}: {}".format(i+1, doc))
        return top_k_relevance

    def test(self, limit:int=10):
        """
        Test the rerank model (passing BM25 results to bi-encoder) on a QnA dataset.
        :param limit: int
        :return: None
        """

        assert limit <= self.bm25_limit, f"Limit must be less than or equal to the BM25 limit: {self.bm25_limit}"
        
        logger.info(f"Limit (k): {limit}")
        rerank_results = {}
        for question_id, result in tqdm(self.full_results.items(), desc="Reranking"):
            split_ids = [split["split_id"] for split in result[:(limit*20)]]
            
            # Retrieve query and splits embeddings
            query_embedding = client.retrieve(collection_name='qna_embedding_legal_1', ids=[question_id], with_payload=False, with_vectors=True)[0].vector
            retrieved_results = client.retrieve(collection_name='embedding_legal_1', ids=split_ids, with_payload=False, with_vectors=True)
            split_embeddings = [result.vector for result in retrieved_results]
            
            # Compute similarity with query and rerank
            query_embedding = np.array(query_embedding).reshape(1, -1)
            split_embeddings = np.array(split_embeddings)
            similarities = cosine_similarity(query_embedding, split_embeddings).flatten()
            top_ids = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
            
            rerank_results[question_id] = [result[i] for i in top_ids]

        # -------------------- Save retrieval results and calculate metrics --------------------
        os.chdir(self.root_dir)
        if not os.path.exists("models/reranker/output"):
            os.makedirs("models/reranker/output")
            logger.info("Created directory: models/reranker/output")
        
        full_retrieval_results_path = f"models/reranker/output/{limit}_reranker_test_results_full.json"
        truncated_retrieval_results_path = f"models/reranker/output/{limit}_reranker_test_results_truncated.json"
        distinct_retrieval_results_path = f"models/reranker/output/{limit}_reranker_test_results_distinct.json"

        true_results_path = "data/true_results/true_test_results.json"

        # ---------- Save full retrieval results ----------
        with open(full_retrieval_results_path, "w") as f:
            json.dump(rerank_results, f, indent=4)
        
        # ---------- Truncate retrieval results ----------
        truncated_results = {}
        for question_id, docs in rerank_results.items():
            truncated_results[question_id] = [doc["law_article_id"] for doc in docs]
        with open(truncated_retrieval_results_path, "w") as f:
            json.dump(truncated_results, f, indent=4)
        
        # ---------- Distinct-article retrieval results ----------
        distinct_results = {}
        for question_id, docs in rerank_results.items():
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

import pandas as pd
import torch
import json
import numpy as np
import time
from code.bm25.bm25_utilities import BM25Utilities
from code.encoder.encoder_utilities import EncoderUtilities
from code.reciprocal_rank_fusion.reciprocal_rank_fusion_utilities import calculate_rrf_ranking
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class CorpusDataset:
    def __init__(self,
                 corpus_df,
                 device, 
                 url = 'https://4d31c99a-9390-4174-8547-167526f0138b.europe-west3-0.gcp.cloud.qdrant.io:6333',
                 api_key = 'WcOwiXDWcFcKJBXkI2zH9LHQ2he0npQNkYTEmS84UGx4kcLwRbMVKg'):
        self.url =url
        self.api_key = api_key
        self.device = device
        self.collection_name = 'embedding_legal_1'
        self.corpus_df = corpus_df
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder').to(self.device)

    def encode(self, text):
        text =[text].to(self.device)
        return self.model.encode(text)[0]
    
    def query(self, segmented_question, topk=10):
        wait = 0.1
        while True:
            try:
                results = self.client.search(
                            collection_name = self.collection_name,
                            query_vector=self.encode(segmented_question),
                            limit=topk,
                        )
                break
            except:
                time.sleep(wait)
                wait*=2
        
        content_results = {}
        for point in results:
            law_id = point.payload['law_article_id']['law_id']
            article_id = point.payload['law_article_id']['article_id']
            key = law_id + '%' + article_id
            if key not in content_results:
                content_results[key] = [point.id]
            else:  
                content_results[key].append(point.id)
        return content_results
    
    def get_id(self, point_id):
        law_article_id = self.corpus_df.iloc[point_id]['law_article_id']
        law_article_id = eval(law_article_id)
        return law_article_id['law_id'] +'%'+ law_article_id['article_id']
    

class HybridSearch:
    def __init__(self, query_df, corpus_df, topk=[1,10,100]):
        self.query_df = query_df
        self.corpus_df = corpus_df
        self.corpus =  corpus_df['segmented_title_content'].tolist()
        self.topk = topk
        self.bm25_util = BM25Utilities(bm25_text=self.corpus)
        self.corpus_util = CorpusDataset(self.corpus_df)

    def get_bm25_rank(self, query, topk):
        return self.bm25_util.get_bm25_ranking(query=query)[:topk]
    
    def get_cosine_rank(self, query, topk):
        rank = self.corpus_util.query(query, topk=topk)
        res = []
        for i in rank.values():
            res.extend(i)
        return np.array(res)
    
    def get_rrf_rank(self, query, topk):
        bm25_rank = self.get_bm25_rank(query, topk)
        cosine_rank = self.get_cosine_rank(query, topk)
        res = calculate_rrf_ranking(bm25_rank, cosine_rank)
        return res
    
    def get_rerank(self, rrf_rank, bm25_rank):
        rrf_rank = [int(i) for i in rrf_rank]
        rerank = bm25_rank[rrf_rank][::-1]
        return rerank
    
    def get_law_article_id(self, rerank):
        res = []
        for point_id in rerank:
            law_article_id = self.corpus_util.get_id(point_id)
            if law_article_id not in res:
                res.append(law_article_id)
        return res
        
    def test(self):
        for k in self.topk:
            res={}
            for i in range(len(self.query_df)):
                query = self.query_df['segmented_question'][i]
                query_id  = self.query_df['question_id'][i]
                bm25_rank = self.get_bm25_rank(query, k)
                cosine_rank = self.get_cosine_rank(query, k)
                rrf_rank = self.get_rrf_rank(query, k)
                rerank = self.get_rerank(rrf_rank, bm25_rank)
                law_article_id = self.get_law_article_id(rerank)
                res[query_id] = law_article_id
            with open('result_'+str(k)+'.json', 'w') as f:
                json.dump(res, f)

    
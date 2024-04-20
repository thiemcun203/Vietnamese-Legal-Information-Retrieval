import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
from qdrant_client import QdrantClient
from qdrant_client.models import *
from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer, AutoModel
import os, json, sys
from dotenv import load_dotenv
sys.path.append(os.getcwd())
from metrics import *
dotenv_path = os.getcwd() + '/.env'
load_dotenv(dotenv_path)

class BiEncoder:
    def __init__(self, 
                 url = os.environ["BKAI_URL"],
                 api_key = os.environ["BKAI_APIKEY"],
                 collection_name = os.environ["BKAI_COLLECTION_NAME"],
                 old_checkpoint = 'bkai-foundation-models/vietnamese-bi-encoder',
                 tunned_checkpoint = '/kaggle/input/checkpoint-1/best_checkpoint.pt',
                 tunned = False,
                ):
        #-----------Setup connection------#
        self.url = url
        self.apikey = api_key
        self.client = QdrantClient(
            url=self.url, 
            api_key=self.apikey,
        )
        self.collection_name = collection_name
        
        #-------Setup device--------#
        if torch.cuda.is_available():
            print("Using GPU")
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_built():
            print("Using MPS")
            self.device = torch.device("mps")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        #-------Model--------#
        self.tokenizer = AutoTokenizer.from_pretrained(old_checkpoint)
        self.model = AutoModel.from_pretrained(old_checkpoint).to(self.device)
        if tunned == True:
            checkpoint = torch.load(tunned_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)

    def encode(self, segmented_questions):
        encoded_input = self.tokenizer(segmented_questions, padding=True, truncation=True, return_tensors='pt', max_length=256)
        # Compute token embeddings
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def query(self, segmented_question = 'khái_niệm quỹ đại_chúng', topk = 10):
        wait = 0.1
        while True:
            try:
                results = self.client.search(
                            collection_name = self.collection_name,
                            query_vector=self.encode([segmented_question])[0],
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
        
    def query_lst(self, segmented_questions = ['khái_niệm quỹ đại_chúng', "hồ_sơ thành_lập quán karaoke bao_gồm những gì ?"], topk = 10):
        vectors = self.encode(segmented_questions)
        search_queries = [SearchRequest(vector=vector, limit=topk, with_payload = True) for vector in vectors]
        results = self.client.search_batch(collection_name=self.collection_name, requests=search_queries)
        return results
    
    def query_to_test(self, topk = 10, limit = (None, None), results_path = 'data/retrieval_results/valid_retrieval_results.json', qna_path = 'data/rdr_data/rdr_valid_qna.csv', batch_size = 4):
        class CorpusDataset(Dataset):
            def __init__(self, dataframe):
                self.dataframe = dataframe

            def __len__(self):
                return len(self.dataframe)

            def __getitem__(self, idx):
                row = self.dataframe.iloc[idx]
                return row['question_id'], row['relevant_articles'], row['segmented_question']
                
        test_df = pd.read_csv(qna_path) if limit[0] is None else pd.read_csv(qna_path)[limit[0] : limit[1]]
        dataset = CorpusDataset(test_df)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        retrieval_results = {}
        for batch in dataloader:
            ques_ids, law_article_ids, questions = batch
            vectors = self.encode(questions)
            search_queries = [SearchRequest(vector=vector, limit=topk, with_payload = True) for vector in vectors]
            wait = 0.1
            while True:
                try:
                    results = self.client.search_batch(collection_name=self.collection_name, requests=search_queries)
                    for qid, points in zip(ques_ids, results):
                        retrieval_results[qid] = list([p.payload['law_article_id']['law_id'] + '%' + p.payload['law_article_id']['article_id'] for p in points])
                    break
                except:
                    time.sleep(wait)
                    wait*=2
        with open(results_path, 'w', encoding='UTF-8') as f:
            json.dump(retrieval_results, f, indent=4)

        return retrieval_results
    
if __name__ == "__main__":
    biencoder = BiEncoder(
                 url = os.environ["BKAI_URL"],
                 api_key = os.environ["BKAI_APIKEY"],
                 collection_name = os.environ["BKAI_COLLECTION_NAME"],
                 old_checkpoint = 'bkai-foundation-models/vietnamese-bi-encoder',
                 tunned_checkpoint = '/kaggle/input/checkpoint-1/best_checkpoint.pt',
                 tunned = False,
                )

    rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
    question = 'khái niệm quỹ đại chúng'
    segmented_question = " ".join(rdrsegmenter.tokenize(question)[0])
    print(segmented_question)
    print(biencoder.query(segmented_question = segmented_question, topk = 10))
    
    # print(biencoder.query_lst(segmented_questions = ['khái niệm quỹ đại chúng', "hồ sơ thành lập quán karaoke bao gồm những gì ?"], topk = 10))
    
    # biencoder.query_to_test(topk = 10, limit = (0, 1), results_path = os.getcwd() + '/data/retrieval_results/valid_retrieval_results.json', qna_path = os.getcwd() + '/data/rdr_data/rdr_valid_qna.csv', batch_size = 4)
    
    # tester = RetrievalMetrics(retrieval_results=os.getcwd() + '/data/retrieval_results/valid_retrieval_results.json', true_results=os.getcwd() + '/data/true_results/true_valid_results.json')
    # recall = tester.get_recall_at_k(k=10)
    # print("Recall@10: ", recall)

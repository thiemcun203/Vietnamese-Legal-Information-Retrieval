import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
from qdrant_client import QdrantClient
from qdrant_client.models import *
from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer, AutoModel
import ast
import os, json, sys
from dotenv import load_dotenv
sys.path.append(os.getcwd())
from models.utils.metrics import *
from models.utils.process import *
dotenv_path = os.getcwd() + '/.env'
load_dotenv(dotenv_path)
df_id = pd.read_csv(os.getcwd() + '/data/corpus/legal_corpus_id.csv')
import time
class BiEncoder:
    def __init__(self, 
                 url = os.environ["BKAI_URL"],
                 api_key = os.environ["BKAI_APIKEY"],
                 collection_name = os.environ["BKAI_COLLECTION_NAME"],
                 old_checkpoint = 'bkai-foundation-models/vietnamese-bi-encoder',
                 tunned_checkpoint = '/kaggle/input/checkpoint-1/best_checkpoint.pt',
                 tunned = False,
                 model_name = 'gpt-3.5-turbo',
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
            # self.device = torch.device('cuda')
            self.device = torch.device("cpu")
        elif torch.backends.mps.is_built():
            print("Using MPS")
            # self.device = torch.device("mps")
            self.device = torch.device("cpu")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        #-------Model--------#
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(old_checkpoint)
        self.model = AutoModel.from_pretrained(old_checkpoint).to(self.device)
        if tunned == True:
            checkpoint = torch.load(tunned_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
        self.sorry_vector = self.encode([tokenize("xin lỗi, tôi không biết, không thể trả lời câu hỏi này, không thể trả lời câu hỏi cá nhân hoặc không liên quan đến pháp luật")])[0]

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
    
    def query(self, df,  question = 'khái_niệm quỹ đại_chúng', topk = 10, LLM = True):
        t = 0
        embed = self.encode([tokenize(question.lower())])[0]
        while True and t < 3:
            wait = 0.1
            while True:
                try:
                    print('Searching...')
                    results = self.client.search(
                                collection_name = self.collection_name,
                                query_vector=embed,
                                limit=topk + t*topk,
                                with_vectors=True,
                            )
                    results = results[t*topk:topk + t*topk]
                    break
                except:
                    time.sleep(wait)
                    wait*=2
                    
                    
            response = ""
            appear = []
            num_docs = 2
            seed = 0
            while LLM:
                context = ''
                with open(os.getcwd() + '/data/corpus/legal_corpus.json', 'r') as f:
                    law_articles = json.load(f)
                
                id_lst = []
                for point in results:
                    law_id, article_id = point.payload['law_article_id']['law_id'], point.payload['law_article_id']['article_id']
                    for law in law_articles:
                        if law['law_id'] == law_id:
                            for article in law['articles']:
                                if article['article_id'] == article_id:
                                    ap = f'{law_id}%{article_id}'
                                    if ap not in appear :
                                        if len(appear[seed:]) < num_docs:
                                            appear.append(ap)
                                            context += f'''Nghị định {law_id} - {article_id}
{article['title']}                  
{article['text']}\n'''    
                                            id_lst += df_id.loc[df_id['law_article_id'] == ap, 'index'].tolist()

                num_try = 0
                print('---------------------Documents------------------')
                print(context)
                if context == '':
                    response = "Tôi không biết."
                    break
                while True:
                    try:
                        context_lst = [i for i in context.split('Nghị định') if i != ''][:num_docs]
                        context = 'Nghị định'.join(context_lst)
                        f_prompt = prompt.format(context, question)
                        response = get_response(f_prompt, model_name = self.model_name)
                        break
                    except Exception as e:
                        if 'maximum context length' in str(e):
                            num_docs -= 1
                            print('retrying...')
                            num_try += 1
                            if num_try == 3 or num_docs == 0:
                                response = "Tôi không biết."
                                break
                            
                if "không biết" in response or 'Xin lỗi' in response:
                    seed +=2
                    # id_lst = []
                    continue
                break
                    
            print(response)
            print(id_lst)
            print(appear)
            if "không biết" not in response and 'Xin lỗi' not in response:
                break
            if context == '': #can be comment for retry
                break
            t+=1
            
            
        related_question = self.client.search(
                                collection_name = 'qna_embedding_legal_1',
                                query_vector=embed,
                                limit=60,
                                with_payload=True,
                            )
        ques_lst = ["".join(point.id.split("-")) for point in related_question]
        ques_lst = [
            df.loc[df['question_id'] == id, 'question'].iloc[0]
            for id in ques_lst  if not df.loc[df['question_id'] == id, 'question'].empty
        ][:15]
        
        return get_rank(self, self.client, question, response, results, f_prompt, df_id,  id_lst, self.sorry_vector, limit = topk, rerank = LLM), response, ques_lst
        
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
                 url = os.environ["SELF_HOST_URL"],
                 api_key = os.environ["SELF_HOST_APIKEY"],
                 collection_name = os.environ["BKAI_COLLECTION_NAME"],
                 old_checkpoint = 'bkai-foundation-models/vietnamese-bi-encoder',
                 tunned_checkpoint = '/kaggle/input/checkpoint-1/best_checkpoint.pt',
                 tunned = False,
                )

    # rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
    # question = 'Điểm tóan thi được mấy điểm?'
    # segmented_question = " ".join(rdrsegmenter.tokenize(question)[0])
    # segmented_question = 'thông_tư này hướng_dẫn tuần_tra , canh_gác bảo_vệ đê_điều trong mùa lũ đối_với các tuyến đê sông được phân_loại , phân_cấp theo quy_định tại điều 4 của luật đê_điều .'
    # segmented_question = 'Mức phạt khi điều khiển xe ô tô không chú ý quan sát gây tai nạn giao thông' #'Không tuân thủ đúng quy trình, quy chuẩn kỹ thuật trong kiểm định xe cơ giới bị phạt bao nhiêu tiền?'#
    # segmented_question = 'Trách nhiệm của Ngân hàng Nhà nước về quản lý và phát triển công nghiệp an ninh được quy định như thế nào?'
    # segmented_question = 'Lập danh mục thủ tục hành chính ưu tiên thực hiện trên môi trường điện tử được thực hiện thế nào?'
    # segmented_question = 'Tự ý di chuyển vị trí biển báo “khu vực biên giới” bị phạt thế nào?'
    # segmented_question = 'Mức phạt khi điều khiển xe ô tô không lắp đủ bánh lốp theo nghị định 100/2019/NĐ-CP?'
    # segmented_question = 'Chủ tàu cá sử dụng tàu cá có chiều dài là 18.5 m khai thác thủy sản mà không có Giấy phép khai thác thủy sản sẽ bị xử phạt như thế nào?'
    segmented_question = 'Thực hiện không đúng quy định về phát bưu gửi bị phạt bao nhiêu?'
    df = pd.read_csv(os.getcwd() + '/data/qna/best_test_qna.csv')
    print(segmented_question)
    t1 = time.time()
    # for i in range(4):
    print(biencoder.query(df,question = segmented_question, topk = 10))
    # print(biencoder.query_lst())
    t2 = time.time()
    print('time', t2-t1)
    
    # print(biencoder.query_lst(segmented_questions = ['khái niệm quỹ đại chúng', "hồ sơ thành lập quán karaoke bao gồm những gì ?"], topk = 10))
    
    # biencoder.query_to_test(topk = 10, limit = (0, 1), results_path = os.getcwd() + '/data/retrieval_results/valid_retrieval_results.json', qna_path = os.getcwd() + '/data/rdr_data/rdr_valid_qna.csv', batch_size = 4)
    
    # tester = RetrievalMetrics(retrieval_results=os.getcwd() + '/data/retrieval_results/valid_retrieval_results.json', true_results=os.getcwd() + '/data/true_results/true_valid_results.json')
    # recall = tester.get_recall_at_k(k=10)
    # print("Recall@10: ", recall)

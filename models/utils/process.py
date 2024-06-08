import re, string, openai, os, sys
from pyvi.ViTokenizer import tokenize
from rank_bm25 import BM25Okapi
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import *
import re
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.getcwd())
from gui.utils import *
dotenv_path = os.getcwd() + '/.env'
import time
load_dotenv(dotenv_path)
openai.api_key  = os.environ["OPENAI_APIKEY"]

with open(os.getcwd()+'/data/stopwords.txt', 'r', encoding="utf-8") as f:
    list_stopwords = f.read().splitlines()

sorry = "Xin lỗi, em chưa có thông tin và không thể trả lời câu hỏi này"
def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub(r'(\s)+', r'\1', text)
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower().strip()

def process_text(text):
    text = clean_text(text)
    text = tokenize(text)
    text = normalize_text(text)
    return text

def remove_stopwords(text, stopwords = list_stopwords):
    return ' '.join([word for word in text.split() if word not in stopwords])

def get_embedding(text, model="text-embedding-3-large"):
    return openai.Embedding.create(input=text, model=model)['data'][0]['embedding']

def get_response(pr):
    # model_name = "gpt-3.5-turbo"
    model_name = "gpt-4"
    retries= 2
    while retries > 0:    
        try: 
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "system", "content": "Bạn là một luật sư chuyên nghiệp ở Việt Nam"},
                          {"role": "user", "content": pr}],
                temperature=0.5,
                max_tokens = 600,
                request_timeout=120,
            )
            return response.choices[0]["message"]["content"]
        except Exception as e:   
            print(e)
            if "context length" not in str(e) and "max_tokens" not in str(e): 
                print(e)   
                print('Timeout error, retrying...')    
                retries -= 1    
                time.sleep(5)   
            else:
                print(e)
                raise e
    return None

prompt = '''Bạn là một luật sư chuyên nghiệp tại Việt Nam, được giao nhiệm vụ trả lời các câu hỏi thường gặp (FAQs) của khách hàng về pháp luật Việt Nam dựa trên thông tin đã cho. 
Vui lòng sử dụng, thu thập và suy luận dựa trên kiến thức trong thông tin sau để trả lời câu hỏi của người dùng. 
Hãy trả lời chính xác, ngắn gọn và đúng trọng tâm, không dài dòng quá mức cần thiết.
Nếu thông tin không đủ để trả lời câu hỏi, hãy nói "Tôi không biết."  và không giải thích gì thêm.
Thông tin pháp lý liên quan:
{0}
Câu hỏi của người dùng:
{1}'''

# Function to extract numbers from text
def extract_numbers(text):
    return re.findall(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?', text)

# Function to tokenize Vietnamese text
def tokenize_vietnamese(text):
    text = process_text(text)
    tokens = [token.strip().lower() for token in text.split()]
    return tokens

def get_rank(model, client, question, response, results, prompt, df_id, id_lst, limit=10, rerank=True) -> str:
    if id_lst == []:
        return {}
    embedding = model.encode([tokenize(response)])[0]
    print(tokenize(response))
    
    def replace_fines(child_str, parent_str):
        # Step 1: Extract monetary amounts from the parent string
        amounts = re.findall(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)? đồng', parent_str)

        # Remove ' đồng' from amounts
        amounts = [amount.replace(' đồng', '') for amount in amounts]

        # Step 2: Replace monetary amounts in the parent string with the placeholder
        # modified_parent_str = re.sub(r'\d{1,10}(?:[.,]\d{10})*(?:[.,]\d+)? đồng', '< mức phạt tiền >', parent_str)

        # Step 3: Match the modified parent string with the child string
        modified_child_str = child_str
        for amount in amounts:
            modified_child_str = modified_child_str.replace('< mức phạt tiền >', amount + " đồng", 1)

        return modified_child_str

    if rerank:
        print('Searching 2...')
        results = client.retrieve(collection_name=os.environ["BKAI_COLLECTION_NAME"], ids=[id for id in id_lst], with_vectors=True, with_payload=True)

    data = {
        'vector': [point.vector for point in results],
        'content': [df_id['content'].iloc[point.id] for point in results],
        'id': [point.id for point in results],
        'law_article_id': [point.payload['law_article_id']['law_id'] + '%' + point.payload['law_article_id']['article_id'] for point in results],
    }

    df = pd.DataFrame(data)
    start = 0
    lst = []
    
    print('---------------------content------------------')
    for val in df['content']:
        lst.append(replace_fines(val, prompt[start:]))
        print(lst[-1])
        _, end = find_index(prompt[start:], val)
        start += end
        
    df['content'] = lst
    tokenized_docs = [tokenize_vietnamese(remove_stopwords(tokenize(con))) for con in df['content']]

    bm25 = BM25Okapi(tokenized_docs)
    
    def search(response):
        tokenized_query = tokenize_vietnamese(remove_stopwords(process_text(response)))
        scores = bm25.get_scores(tokenized_query)
        return scores

    scores = search(response)
    
    # Extract numbers from the response
    response_numbers = extract_numbers(response)
    
    # Adjust BM25 scores based on number matching
    def adjust_bm25_score(content, score):
        content_numbers = extract_numbers(content)
        matches = set(response_numbers) & set(content_numbers)
        if matches:
            return score * (1 + 0.2 * len(matches))  # Increase score by 10% for each match
        return score

    df['bm25'] = [adjust_bm25_score(con, score) for con, score in zip(df['content'], scores)]
    df['bm25'] = [score/max(df['bm25']) if max(df['bm25']) > 0 else 0 for score in df['bm25']]
    # df['bm25'] = [score/max(scores) if max(scores) > 0 else 0 for score in scores]
    
    df['similarities'] = df['vector'].apply(lambda x: cosine_similarity([x], [embedding])[0][0])
    df['similarities'] = df['similarities'].apply(lambda x: x/max(df['similarities']) if max(df['similarities']) > 0 else 0)
    
    df['score'] = 0.65 * df['similarities'] + 0.35 * df['bm25']
    
    if rerank:
        df = df.sort_values('score', ascending=False).head(limit) #
    
    print('sorted', df[['id', 'similarities', 'bm25', 'score','content']])
    

    content_results = {}
    k  = 0 
    for i, r in df.iterrows():
        key = r['law_article_id']
        if key not in content_results:
            content_results[key] = [r['content']]
            k+=1
        else:
            if r['score']/df['score'].iloc[k-1] > 0.89:
                content_results[key].append(r['content'])
                k+=1
            else:
                break
        
    
    return content_results


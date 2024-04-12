
from pyvi.ViTokenizer import tokenize
import re 
import pandas as pd 
import numpy as np 
import torch 
from tqdm.auto import tqdm
tqdm.pandas()

from sentence_transformers import SentenceTransformer
import cupy as cp  

model_device = "cuda:0" if torch.cuda.is_available() else "cpu" 
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=model_device)

def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def word_segment(sent):
    sent = tokenize(sent.encode('utf-8').decode('utf-8'))
    return sent

def encode(lst = [], batch_size = 32, convert_to_tensor=True):
    encoded_vectors = model.encode(lst, convert_to_tensor=True, device=device, batch_size=batch_size, show_progress_bar=True)
    encoded_vectors_cpu = cp.asnumpy(encoded_vectors)  # Convert to NumPy array
    vector_arr = [np.array(arr) for arr in encoded_vectors.cpu().detach().numpy()]
    return vector_arr

model = None
cos_sim_corpus = pd.read_hdf("corpus_2.h5") 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

atts = cos_sim_corpus.filter(['id', 'vector'], axis = 1) 
atts['vector'] = atts['vector'].progress_apply(torch.tensor) 
vector_stacks = torch.stack(list(atts['vector'])).to(device)

del cos_sim_corpus

cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps = 1e-6).to(device)

def get_cosine_similarity_based_ranking(query, atts=atts, vector_stack: torch.Tensor = vector_stacks, cos_sim: torch.nn.CosineSimilarity = cosine_similarity, model = model, batch = True):
    if not batch:  
        query = clean_text(query)
        query = word_segment(query)
        embeddings = model.encode(query) 
        score = cos_sim(torch.tensor([embeddings]).to(device), vector_stack)
        score = score.cpu().detach().numpy()
        atts['score'] = score.T
        atts = atts.sort_values('score', ascending = False)

        return atts["id"].to_numpy()
    
    else: 
        embeddings = torch.tensor(query).to(device)
        embeddings = torch.stack([embeddings])
        score = cos_sim(embeddings, vector_stack)
        score = score.cpu().detach().numpy()
        atts['score'] = score.T 
        atts = atts.sort_values('score', ascending = False)

        return atts["id"].to_numpy()
    
def get_top_relevance_ids_in_batches(bm25_query, cos_sim_query, ranking, bm25_ranking = None, cos_sim_ranking = None, unique = True): 
    if ranking == "cos_sim": 
        top_ids = get_cosine_similarity_based_ranking(query=cos_sim_query,  batch = True) 
    return top_ids
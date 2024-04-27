import pandas as pd
from code.preprocess.preprocess_functions import * 
from tqdm import tqdm
tqdm.pandas() 
from sentence_transformers import SentenceTransformer
import torch 
import numpy as np 

def encode(model, lst = [], batch_size = 32): 
        encoded_vectors = model.encode(lst, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True) 
        vector_arr = [np.array(arr) for arr in encoded_vectors.cpu().detach().numpy()]
        return vector_arr

def preprocess_for_encoder(batch_size = 32):
    print("Loading model...")
    model_device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=model_device) 

    print("Preprocessing corpus for encoder...")
    path_to_corpus = "input/rdrsegmenter_legal_corpus.csv"
    corpus = pd.read_csv(path_to_corpus) 
    corpus['segmented_title_content'] = corpus.progress_apply(lambda x: encoder_preprocess(x.segmented_title_content), axis = 1) 

    print("Encoding corpus...") 
    corpus['encoded_content'] = encode(model, list(corpus['segmented_title_content']), batch_size=batch_size)
    corpus.to_hdf("midpoints/encoder_corpus.h5", key="encoder_corpus")
  
    print("Preprocessing qna for encoder...")
    path_to_qna = "input/rdrsegmenter_testqna.csv"
    qna = pd.read_csv(path_to_qna)
    qna['segmented_question'] = qna.progress_apply(lambda x: encoder_preprocess(x.segmented_question), axis = 1)

    print("Encoding qna...")
    qna_questions = list(qna['segmented_question'])
    qna['encoded_question'] = encode(model, qna_questions, batch_size=batch_size) 
    qna.to_hdf("midpoints/encoder_test_qna.h5", key="encoder_qna")
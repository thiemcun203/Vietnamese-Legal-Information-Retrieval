from code.encoder.encoder_utilities import EncoderUtilities
from tqdm import tqdm
import pandas as pd
import torch
tqdm.pandas()

def encoder_get_top_k_relevance(embedding, encoder_utilities):
    top_ids = encoder_utilities.get_encoder_ranking(embedding, preprocessed = True)
    top_k_relevance = encoder_utilities.get_top_k_relevance(top_ids = top_ids)
    return top_k_relevance

def encoder_process(): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder_corpus = pd.read_hdf("midpoints\\encoder_corpus.h5")
    encoder_utilities = EncoderUtilities(device=device, encoder_corpus=encoder_corpus, look_up=True, limit = 10)

    encoder_test_qna = pd.read_hdf("midpoints\\encoder_test_qna.h5")
    print("Getting top Encoder relevances for qna dataset...")
    encoder_test_qna['top_relevance'] = encoder_test_qna.progress_apply(lambda x: encoder_get_top_k_relevance(x.encoded_question, encoder_utilities=encoder_utilities), axis = 1)
    encoder_test_qna = encoder_test_qna.drop(columns = ["encoded_question"])
    encoder_test_qna.to_csv("output\\encoder_test_qna_output.csv")
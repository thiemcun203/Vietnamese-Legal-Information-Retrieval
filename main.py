from code.bm25.bm25_preprocess import preprocess_for_bm25
from code.bm25.bm25_process import bm25_process
from code.encoder.encoder_preprocess import preprocess_for_encoder
from code.encoder.encoder_process import encoder_process
from code.reciprocal_rank_fusion.reciprocal_rank_fusion_preprocess import preprocess_for_reciprocal_rank_fusion
from code.reciprocal_rank_fusion.reciprocal_rank_fusion_process import rrf_process  

import os

if __name__ == '__main__':

    if not os.path.exists("midpoints"):
        os.makedirs("midpoints")
    if not os.path.exists("output"):
        os.makedirs("output")
        
    preprocess_for_bm25()
    bm25_process()

    preprocess_for_encoder(batch_size=400)
    encoder_process()
    
    preprocess_for_reciprocal_rank_fusion()
    rrf_process()
 
    print("Done.")
    
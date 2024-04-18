# from code.bm25.bm25_preprocess import preprocess_for_bm25
# from code.bm25.bm25_process import bm25_process
# from code.encoder.encoder_preprocess import preprocess_for_encoder
# from code.encoder.encoder_process import encoder_process
# from code.reciprocal_rank_fusion.reciprocal_rank_fusion_preprocess import preprocess_for_reciprocal_rank_fusion
# from code.reciprocal_rank_fusion.reciprocal_rank_fusion_process import rrf_process 
import pandas as pd

if __name__ == '__main__':
    # preprocess_for_bm25()
    # bm25_process()

    # preprocess_for_encoder(batch_size=400)
    # encoder_process()
    
    # preprocess_for_reciprocal_rank_fusion()
    # rrf_process()

    qna = pd.read_csv("output\\rrf_test_qna_output.csv")
    qna = qna.drop(columns = ["encoded_question"])
    qna.to_csv("hey.csv")
    print("Done.")
    
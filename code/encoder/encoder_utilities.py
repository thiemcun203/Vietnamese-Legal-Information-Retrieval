import torch

class EncoderUtilities:
    def __init__(self, device, encoder_corpus, look_up = True, limit = 10):
        print("EncoderUtilities initializing...")
        self.device = device
        atts = encoder_corpus.filter(['index', 'encoded_content'], axis = 1)
        atts['encoded_content'] = atts['encoded_content'].apply(torch.tensor)
        self.vector_stack = torch.stack(list(atts['encoded_content'])).to(self.device)
        self.indices = atts[["index"]].copy()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps = 1e-6).to(self.device)

        self.look_up = look_up
        if look_up == True:  
            self.corpus_look_up = encoder_corpus['law_article_id']  
    
        self.limit = limit

    def get_encoder_ranking(self, query, preprocessed = True):
        model = None
        if not preprocessed: 
            embedding = model.encode(query)
        else:
            embedding = torch.tensor(query).to(self.device)
        embedding = torch.stack([embedding])
        score = self.cosine_similarity(embedding, self.vector_stack)
        score = score.cpu().detach().numpy()
        indices = self.indices.copy()
        indices['score'] = score.T
        indices = indices.sort_values('score', ascending = False)

        return indices['index'].to_numpy()
    
    def get_top_k_relevance(self, top_ids):

        assert self.look_up == True 
        top_k_results = [self.corpus_look_up[i] for i in top_ids][:self.limit]

        return top_k_results

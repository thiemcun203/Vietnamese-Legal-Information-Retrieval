from functools import reduce
import json

class RetrievalMetrics:
    def __init__(self, retrieval_results:str, true_results:str='true_test_results.json')->None:
        """
        - retrieval_results: str, path to the json file containing the retrieval results
        - true_results: str, path to the json file containing the true relevant articles
        - Both files follow the format:
        {
            "question_id": ["article_id_1", "article_id_2", ...]
        }
        Note that the article_ids are sorted by relevance, i.e., the first article is the most relevant one.
        - question_id is taken as is from the dataset
        - article_id is the law title concatenated with the article number using a % sign, for example: "28/2020/nđ-cp%21"
        """
        with open(retrieval_results, 'r') as f:
            self.retrieval_results = json.load(f)
        with open(true_results, 'r') as f:
            self.true_results = json.load(f)
        self.num_questions = len(self.retrieval_results)
        assert len(set(self.retrieval_results.keys()) & set(self.true_results.keys())) == len(self.retrieval_results)
        
    
    def get_precision_at_k(self, question_id:str=None, k:int=1)->float:
        
        if question_id:
            retrieved_articles = self.retrieval_results[question_id]
            true_relevant_articles = self.true_results[question_id]
            assert k <= len(retrieved_articles)
            
            precision = len(set(retrieved_articles[:k]) & set(true_relevant_articles)) / k
            return precision
        
        else:
            precisions = []
            for question_id in self.retrieval_results:
                precisions.append(self.get_precision_at_k(question_id, k))
            return sum(precisions) / self.num_questions
    
    def get_recall_at_k(self, question_id:str=None, k:int=1)->float:
        
        if question_id:
            retrieved_articles = self.retrieval_results[question_id]
            true_relevant_articles = self.true_results[question_id]
            assert k <= len(retrieved_articles)

            recall = len(set(retrieved_articles[:k]) & set(true_relevant_articles)) / len(true_relevant_articles)
            return recall
        
        else:
            recalls = []
            for question_id in self.retrieval_results:
                recalls.append(self.get_recall_at_k(question_id, k))
            return sum(recalls) / self.num_questions
    
    def get_f_beta_score_at_k(self, question_id:str=None, k:int=1, beta:float=1)->float:
        
        if question_id:
            precision = self.get_precision_at_k(question_id, k)
            recall = self.get_recall_at_k(question_id, k)
            if precision + recall == 0:
                return 0
            f_beta_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            return f_beta_score
        
        else:
            f_beta_scores = []
            for question_id in self.retrieval_results:
                f_beta_scores.append(self.get_f_beta_score_at_k(question_id, k, beta))
            return sum(f_beta_scores) / self.num_questions

    def get_pr_curve(self, question_id:str)->list:
        
        if question_id:
            retrieved_articles = self.retrieval_results[question_id]
            pr_curve = []
            for k in range(1, len(retrieved_articles)+1):
                precision = self.get_precision_at_k(question_id, k)
                recall = self.get_recall_at_k(question_id, k)
                pr_curve.append((precision, recall))
            return pr_curve
    
    def get_interpolated_pr_curve(self, question_id:str=None)->list:
        
        if question_id:
            pr_curve = self.get_pr_curve(question_id)
            interpolated_pr_curve = []
            for i in range(0, 11):
                tick = i/10
                interpolated_precision = max([precision for precision, recall in pr_curve if recall >= tick])
                interpolated_pr_curve.append((interpolated_precision, tick))
            return interpolated_pr_curve
        
        else:
            add = lambda x, y: (x[0] + y[0], x[1])
            avg = lambda x: (x[0]/self.num_questions, x[1])
            interpolated_pr_curves = []
            for question_id in self.retrieval_results:
                interpolated_pr_curves.append(self.get_interpolated_pr_curve(question_id))
            interpolated_pr_curve = list(map(avg, [add(x, y) for (x, y) in zip(*interpolated_pr_curves)]))
            return interpolated_pr_curve
    
    def get_mrr(self)->float:
        
        mrr = 0
        for question_id in self.retrieval_results:
            retrieved_articles = self.retrieval_results[question_id]
            true_relevant_articles = self.true_results[question_id]
            for i, article_id in enumerate(retrieved_articles):
                if article_id in true_relevant_articles:
                    mrr += 1/(i+1)
                    break
        return mrr / self.num_questions

    def get_ap_at_k(self, question_id:str, k:int=1)->float:
        
        retrieved_articles = self.retrieval_results[question_id]
        true_relevant_articles = self.true_results[question_id]
        assert k <= len(retrieved_articles)
        ap = 0
        num_relevant_articles = 0
        for i, article_id in enumerate(retrieved_articles[:k]):
            if article_id in true_relevant_articles:
                num_relevant_articles += 1
                ap += num_relevant_articles / (i+1)
        return ap / len(true_relevant_articles)

    def get_map_at_k(self, k:int=1)->float:
        
        map = 0
        for question_id in self.retrieval_results:
            map += self.get_ap_at_k(question_id, k=k)
        return map / self.num_questions
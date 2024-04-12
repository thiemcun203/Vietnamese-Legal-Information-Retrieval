
import pandas as pd
from pyvi.ViTokenizer import tokenize
import re, os, string
import math
import numpy as np
import pickle

def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()

filename = 'input/stopwords.txt'
list_stopwords = open(filename, 'r').readlines()
list_stopwords = [i.replace(" ", "_") for i in list_stopwords] 

def remove_stopword(text):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)

    return text2

def word_segment(sent):
    sent = tokenize(sent.encode('utf-8').decode('utf-8'))
    return sent

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):
        """
        Fit the various statistics that are required to calculate BM25 ranking
        score using the corpus given.

        Parameters
        ----------
        corpus : list[list[str]]
            Each element in the list represents a document, and each document
            is a list of the terms.

        Returns
        -------
        self
        """
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            tf.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query):
        scores = [self._score(query, index) for index in range(self.corpus_size_)]
        return scores

    def _score(self, query, index):
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)

        return score
    
bm25_corpus = pd.read_csv("corpus.csv")

fp = open("text", "rb")
texts = pickle.load(fp)

bm25 = BM25()
bm25.fit(texts)

def get_bm25_ranking(query, bm25 = bm25, batch = True):
    if not batch:
        query = clean_text(query)
        query = word_segment(query)
        query = remove_stopword(normalize_text(query))
        query = query.split()
        
        scores = bm25.search(query)
        scores_index = np.argsort(scores)
        scores_index = scores_index[::-1]
        return scores_index
    
    else:
        query = query.split() 
        scores = bm25.search(query)
        scores_index = np.argsort(scores)
        scores_index = scores_index[::-1]

        return scores_index
    
def get_top_relevance_ids_in_batches(bm25_query, cos_sim_query, ranking, bm25_ranking = None, cos_sim_ranking = None, unique = True):
    if ranking == "bm25":
        top_ids = get_bm25_ranking(query=bm25_query, batch = True)
    return top_ids
import os
import re, string
from python_rdrsegmenter import load_segmenter

# -------------------- Load stopwords --------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
filename = '/'.join(current_dir.split('/')[:-2]) + '/data/stopwords.txt'
list_stopwords = open(filename, 'r', encoding="utf8").read().splitlines()
list_stopwords = [i.replace(" ", "_") for i in list_stopwords]  

# -------------------- Preprocess functions --------------------
def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()

def remove_stopword(text):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)
    return text2

def word_segment(sent):
    segmenter = load_segmenter()
    sent = segmenter.tokenize(sent.encode('utf-8').decode('utf-8'))
    return sent

# -------------------- Overall preprocess functions --------------------
def bm25_preprocess(query, skip_word_segment=True):
    if skip_word_segment:
        query = remove_stopword(normalize_text(clean_text(query))) 
    else:
        query = remove_stopword(normalize_text(word_segment(clean_text(query)))) 
    return query

def bi_encoder_preprocess(query):
    query = word_segment(clean_text(query))
    return query

from python_rdrsegmenter import load_segmenter
segmenter = load_segmenter() 
import re, string 

def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()

filename = 'input\\stopwords.txt'
list_stopwords = open(filename, 'r', encoding="utf8").read().splitlines()
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
    sent = segmenter.tokenize(sent.encode('utf-8').decode('utf-8'))
    return sent

def bm25_preprocess(query):
    query = remove_stopword(normalize_text(word_segment(clean_text(query)))) 
    return query

def encoder_preprocess(query):
    query = word_segment(clean_text(query))
    return query

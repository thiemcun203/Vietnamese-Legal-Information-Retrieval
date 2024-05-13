import os, json
import pandas as pd

def find_index(text1, text2):
    # Create a 2D array to store lengths of longest common suffixes
    n, m = len(text1), len(text2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Variables to store the length and the end index of the longest common substring in text1
    max_length = 0
    ending_index_text1 = 0
    
    # Fill dp array
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    ending_index_text1 = i - 1
            else:
                dp[i][j] = 0

    # Start index of the longest common substring in text1
    start_index = ending_index_text1 - max_length + 1
    if max_length == 0:
        return -1, -1
    if len(text2) - 10 > max_length:
        return -1, -1
    # Return the start and end indices along with the substring
    return start_index, ending_index_text1

    return start_index, end_index

def find_documents(retrived_id = {'155/2015/tt-btc%2': [144554, 144566], '54/2019/qh14%4': [311550], '96/2020/tt-btc%3': [385427], '54/2019/qh14%110': [312460, 312459, 312467, 312471], '54/2019/qh14%124': [312585], '54/2019/qh14%108': [312451]}):
    """
    Find the documents in the database that match the retrieved documents
    """
    with open(os.getcwd() + '/data/corpus/legal_corpus.json', 'r') as f:
        law_articles = json.load(f)
    splitted_corpus  = pd.read_csv(os.getcwd() + '/data/corpus/splitted_legal_corpus_id.csv')
    
    documents = {}
    for id in retrived_id:
        law_id, article_id = id.split('%')
        documents[id] = {}
        for law in law_articles:
            if law['law_id'] == law_id:
                for article in law['articles']:
                    if article['article_id'] == article_id:
                        documents[id]['full_content'] = f'''{law_id} - {article_id}
{article['title']}
{article['text']}'''    
                        
        annotations = []
        splitted_info = ''
        for doc_id in retrived_id[id]:
            retrieved_split =  splitted_corpus.loc[doc_id]['content']
            splitted_info += f'{retrieved_split}\n'
            start_index, end_index = find_index(documents[id]['full_content'].lower(), splitted_corpus.loc[doc_id]['content'])
            if start_index != -1:
                annotations.append({"start": start_index, "end": end_index, "tag": "Suggestions"})
        documents[id]['splitted_info'] = splitted_info
        documents[id]['annotations'] = annotations
    return documents



if __name__ == "__main__":
    x = find_documents()['155/2015/tt-btc%2']
    from text_highlighter import text_highlighter
    import streamlit as st

    # Basic usage
    text_highlighter(
        text=x['full_content'],
        labels=[("Suggestions", "C0D6E8")],
        # Optionally you can specify pre-existing annotations:
        annotations=x['annotations'],
    )

    # Show the results (in XML format)
    # st.code(result.to_xml())

    # Show the results (as a list)
    # st.write(result)
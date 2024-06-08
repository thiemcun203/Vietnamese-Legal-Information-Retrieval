import copy
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import random, time, re, json
import streamlit as st
from streamlit_feedback import streamlit_feedback
from google.cloud import firestore
from google.oauth2 import service_account
import json, sys, os
sys.path.append(os.getcwd())
from  models.biencoder_model.BiEncoder  import BiEncoder
from models.bm25.bm25_model import BM25
from models.hybrid_model.hybrid import Hybrid
from vncorenlp import VnCoreNLP
from utils import *
from text_highlighter import text_highlighter
import pandas as pd
import numpy as np
from pyvi.ViTokenizer import tokenize
#  python -m streamlit run gui/main.py

# #initialize database for collect user data
@st.cache_resource 
def connect_user():
    key_dict = json.loads(st.secrets["textkey"])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    home = firestore.Client(credentials=creds, project=key_dict["project_id"])
    db = home.collection("user_feedback")
    return db

records = connect_user()

@st.cache_resource 
def init_BiEncoder():
    return BiEncoder(
            url = st.secrets["SELF_HOST_URL"],
            api_key = st.secrets["SELF_HOST_APIKEY"],
            collection_name = st.secrets["BKAI_COLLECTION_NAME"],
            old_checkpoint = 'bkai-foundation-models/vietnamese-bi-encoder',
            tunned_checkpoint = '/kaggle/input/checkpoint-1/best_checkpoint.pt',
            tunned = False,
        )
# @st.cache_resource 
# def init_BM25():
#     bm25 =  BM25(k1=1.2, b=0.65)
#     bm25.fit()
#     return bm25
    
biencoder = init_BiEncoder()
# bm25  = init_BM25()

# @st.cache_resource 
# def init_hybrid():
#     return Hybrid(biencoder, bm25, const = 60, biencoder_rate = 1.75, bm25_rate = 1)

# hybrid = init_hybrid()

@st.cache_data
def load_data():
    df = pd.read_csv(os.getcwd() + '/data/qna/best_test_qna.csv')
    return df
df = load_data()
if 'seed' not in st.session_state:
    st.session_state.seed = np.random.randint(0, 101)
# Randomly pick 3 unique questions
questions_sample = df['question'].sample(n=3, random_state=np.random.RandomState(st.session_state.seed)).tolist()

# Setup memorize the conversation
if 'buffer_memory' not in st.session_state:
    st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=1,return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question in Vietnamese as truthfully as possible using the provided context,
""") #and if the answer is not contained within the text below, say 'Tôi không biết'
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets["apikey"])
conversation = ConversationChain(memory=st.session_state.buffer_memory,prompt=prompt_template, llm=llm, verbose=True)

if 'count' not in st.session_state:
    st.session_state.count = 0
st.session_state.count += 1

if 'fb' not in st.session_state:
    st.session_state.fb = 0

if 'data' not in st.session_state:
    st.session_state.data = (None,None)
    
if 'double' not in st.session_state: #fix double bug
    st.session_state.double = 0

if 'results' not in st.session_state:
    st.session_state.results = {}


# setup UI
st.subheader("LegalBot🤖")
option = st.selectbox(
    'Model Name',
    ('Hybrid-Model', '')) # , 'BM25-Model', 'BKAI-Model'
if option == 'Hybrid-Model':
    model = biencoder
# elif option == 'BM25-Model':
#     model = bm25
# elif option == 'Hybrid-Model':
#     model = hybrid
    
first_message = "Tôi có thể giúp gì cho bạn?"

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": first_message}]

# Display chat messages
for i,message in enumerate(st.session_state.messages): #do not print all messages
    with st.chat_message(message["role"], avatar="😎" if message["role"] == "user" else "🤖"):
        st.write(message["content"])

if prompt := st.chat_input():
    if st.session_state.double + 1 != st.session_state.count:
        st.session_state.double = copy.deepcopy(st.session_state.count)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="😎"):
            st.write(prompt)
            
if st.button(questions_sample[0]):
    prompt = questions_sample[0]
    st.session_state.messages.append({"role": "user", "content": copy.deepcopy(questions_sample[0])})
if st.button(questions_sample[1]):
    prompt = questions_sample[1]
    st.session_state.messages.append({"role": "user", "content": copy.deepcopy(questions_sample[1])})
if st.button(questions_sample[2]):
    prompt = questions_sample[2]
    st.session_state.messages.append({"role": "user", "content": copy.deepcopy(questions_sample[2])})

with st.sidebar:
    # for id in st.session_state.results:
    for doc_key, doc_info in st.session_state.results.items():

        labels = [("rank_1", "#7F5A3E"), ("rank_2", "#DAAC8D"),  ("rank_3", "#FED8B1"),  ("rank_4", "#FDEAD6"),  ("rank_5", "#FFF8F1"),  ("rank_6", "#FFF8F1"),  ("rank_7", "#FFF8F1"),("rank_8", "#FFF8F1"),  ("rank_9", "#FDFDFD"),  ("rank_10", "#FDFDFD")]

        # Displaying the highlighted text
        text_highlighter(
            text=doc_info['full_content'],
            labels=labels,
            annotations=doc_info['annotations'],
        )


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            # rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
            # segmented_question = " ".join(rdrsegmenter.tokenize(prompt)[0])
            # segmented_question = prompt.encode('utf-8').decode('utf-8')
            segmented_question = prompt
            t1 = time.time()
            print('question: ', segmented_question)
            results, response = model.query(question = segmented_question, LLM = True)

            results = find_documents(results)

            # context = ''
            # for id in results:
            #     context += f"{results[id]['full_content']}\n"
            # context = [results[id]['splitted_info'] for id in results]

            # print(context)
            # response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{prompt}")
            st.write(response) 
            
            st.session_state.seed += np.random.randint(0, 1000)
            st.session_state.results = results
            t2 = time.time()
            print("Time: ", t2 - t1)
        
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
    # save user data
    data = {
            "timestamp": time.time(), #dt_object = datetime.fromtimestamp(timestamp)
            "user_message": st.session_state.messages[-2]["content"],
            "bot_message": st.session_state.messages[-1]["content"],
            "context_id":str(results),
            "like": None,
            "feedback": None,
            "feedback_time": None
        }
    
    _ , ref = records.add(data)
    st.session_state.data = (ref.id,data)
    st.experimental_rerun()


if st.session_state.count > 1:
    feedback = streamlit_feedback( #each feedback can only used once
                    feedback_type=f"thumbs",
                    key=f"{st.session_state.fb}",
                    optional_text_label="[Tuỳ chọn] Lý do") #after click, reload and add value for next load
    if feedback:
        st.session_state.messages[-1]["feedback"] = feedback
        st.session_state.fb += 1 #update feedback id
        
        #retrieve desired data from database
        id, data = st.session_state.data
        doc_ref = records.document(id)
        doc_ref.update({"timestamp":data["timestamp"],
                        "user_message": data["user_message"],
                        "bot_message": data["bot_message"],
                        "context_id": data["context_id"],
                        "like": 1 if feedback["score"] == "👍" else 0,
                        "feedback": feedback["text"],
                        "feedback_time": time.time()
                        })


print("Done turn! State: ",st.session_state.count) 
#each action, fb - refresh page is a turn



          
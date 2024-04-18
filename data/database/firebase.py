from google.cloud import firestore
from google.oauth2 import service_account
import streamlit as st
import json
class FirebasePipeline(object):
    def __init__(self, firebase_key: dict, project_id: str, collection_name:str = "user_feedback"):
        # Initialize the app with a service account, granting admin privileges
        self.creds = service_account.Credentials.from_service_account_info(firebase_key)
        self.project_id = project_id
        # Get the database "batdongsan"
        self.database = firestore.Client(credentials=self.creds, project=self.project_id)
        self.collection = self.database.collection(collection_name) 

    def update_item(self,id,item):
        doc_ref = self.database.document(id)
        
        doc_ref = collection.document(id)
        doc = doc_ref.get()
        return item
    
    def get_item(self, id):
        collection = self.database.collection(spider.name[:-6])
        
if __name__ == "__main__":
    def connect_user():
        key_dict = json.loads(st.secrets["textkey"])
        creds = service_account.Credentials.from_service_account_info(key_dict)
        home = firestore.Client(credentials=creds, project=key_dict["project_id"])
        db = home.collection("user_feedback")
        return db

    records = connect_user()


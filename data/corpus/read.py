# import json
# # with open('data/corpus/legal_corpus.json', 'r') as f:
# #     corpus = json.load(f)
# # count_lawid = 0
# # print(len(corpus))
# # sum_art = 0
# # for law in corpus:
# #     sum_art += len(law['articles'])
# # print(sum_art/len(corpus))
# import json

# # Assuming corpus_list contains all the JSON objects
# corpus_list = []

# # Read each JSON object from the file
# with open('data/corpus/1.json', 'r', encoding='utf-8') as f:
#     for line in f:
#         try:
#             json_object = json.loads(line)
#             corpus_list.append(json_object)
#         except json.JSONDecodeError as e:
#             print(f"Error decoding JSON: {e}")
# print(corpus_list[4])
# # Process each JSON object as needed (for example, modifying data in each object)
# # Here we just use the objects as is.

# # Write all JSON objects back to a single file
# with open('data/corpus/1_modified.json', 'w', encoding='utf-8') as f:
#     json.dump(corpus_list, f, indent=4)
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
client = QdrantClient(url="http://kaximgroup.ddns.net:6333", api_key= 'Y3NMMGK3Okzt7rzho88jhmzPZl5MhmAtwFbLthiemcun2037pH8sOlVNtveGcejEl') # Y3NMMGK3Okzt7rzho88jhmzPZl5MreadLG4OJPGRtNV7pH8sOlVNtveGcejEl
# client.create_collection(
#     collection_name="test_collection",
#     vectors_config=VectorParams(size=4, distance=Distance.DOT),
# )
from qdrant_client.models import PointStruct

# operation_info = client.upsert(
#     collection_name="test_collection",
#     wait=True,
#     points=[
#         PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
#         PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
#         PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": ""}),
#         PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New Y"}),
#         PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
#         PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
#     ],
# )

# print(operation_info)
search_result = client.search(
    collection_name="test_embedding_legal_1", query_vector=[0.7]*768, limit=3
)
results_q = client.scroll(
    collection_name="qna_embedding_legal_1",
    with_vectors = True,
    scroll_filter=models.Filter(
        must=[
            models.HasIdCondition(has_id=[process_id(train_df['question_id'].iloc[id]) for id in range(i,i+batch)]),
            ],
        ),
    )[0]

print(search_result)
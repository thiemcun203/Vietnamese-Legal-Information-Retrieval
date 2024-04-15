# Legal-Information-Retrieval

## How to use metrics.py

Ensure that the output to whatever model you used is a json file of the format
```
{
    "question_id": ["article_id_1", "article_id_2", ...]
}
```
where
- question_id is taken as is from the dataset
- article_id is the law title concatenated with the article number using a % sign, for example: "28/2020/nđ-cp%21"

Pass that file, along with ``true_results.json`` here to initialize a ``RetrievalMetrics`` object. The methods available have been tested and should be easy to understand.
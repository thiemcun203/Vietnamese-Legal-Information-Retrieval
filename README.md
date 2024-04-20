# Legal-Information-Retrieval

## How to use ``metrics.py``

Ensure that the output to whatever model you used is a json file of the format
```
{
    "question_id": ["article_id_1", "article_id_2", ...]
}
```
where
- question_id is taken as is from the dataset
- article_id is the law title concatenated with the article number using a % sign, for example: "28/2020/nđ-cp%21"
Example
```
{
    "8e2cfe626cebf209f94e0db8f147960c":[
        "28/2020/nđ-cp%21",
        "64/2020/qh14%97",
        "26/2019/nđ-cp%28",
        "59/2020/qh14%180",
        "01/2016/qh14%12]"
    ],
    "0a32724630653580cc90c77bcf552baf":[
        "28/2020/nđ-cp%21",
        "64/2020/qh14%97",
        "26/2019/nđ-cp%28",
        "59/2020/qh14%180",
        "01/2016/qh14%12]"
    ]
}
```

Pass that file, along with ``true_results.json`` to initialize a ``RetrievalMetrics`` object. The methods available have been tested and should be easy to understand.

## How to run BM25

Add `rdrsegmenter_legal_corpus.csv` to `input/` and then

Run `main.py`:

```python main.py```

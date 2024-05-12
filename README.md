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

## How to run GUI
1. Make sure you downloaded all data from this link: https://drive.google.com/drive/folders/1J0GdSm2bY7GM-MCUQtPnceLrRhtGlrTt?usp=sharing
2. Download `secrets.toml` and `.env` also from above link to configurate private key for application. Put `secrets.toml` into `.streamlit` folder which is same level with other folder like: `gui`, `data` or `.env` file
3. Create virtual enviroment
4. Run setup.sh to setup and download all necessary packages by typing the following commands to terminal:
   1. `chmod +x setup.sh`
   2. `./setup.sh`
5. Start to use GUI: `streamlit run gui/main.py`

## How to use vinhdoan's BM25
1. Add `rdrsegmenter_legal_corpus.csv` and `rdrsegmenter_testqna` (on `dtv/bm25` on Google Drive) to `input/`
2. Instantiate `BM25Preprocessor`
```
preprocessor = BM25Preprocessor()
preprocessor.preprocess_corpus("/input/rdrsegmenter_legal_corpus.csv", skip_word_segment=True)
preprocessor.preprocess_qna("/input/rdrsegmenter_testqna.csv", skip_word_segment=True)
```
(Optional)
Place `bm25_full_corpus.h5`, `bm25_only_text`, `bm25_test_qna.h5` in `bm25/processed_data` to avoid having to preprocess.
3. Instantiate `BM25`
```
bm25 = BM25(k1=1.2, b=0.65)
bm25.fit()
```
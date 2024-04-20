#!/bin/bash
#chmod +x setup.sh

# Install Python packages
pip install -r requirements.txt

# Download VnCoreNLP and models
mkdir -p vncorenlp/models/wordsegmenter
wget --no-check-certificate https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar -O vncorenlp/VnCoreNLP-1.1.1.jar
wget --no-check-certificate https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab -O vncorenlp/models/wordsegmenter/vi-vocab
wget --no-check-certificate https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr -O vncorenlp/models/wordsegmenter/wordsegmenter.rdr
# mv VnCoreNLP-1.1.1.jar vncorenlp/ 
# mv vi-vocab vncorenlp/models/wordsegmenter/
# mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
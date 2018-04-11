# Yiyu Sentiment Analysis Project
Created by Yuelin Hu and Yixin Qian

## Dependencies
1. Python >= 3.6
2. pands >= 0.21.1
3. numpy >= 1.13.3
4. sklearn >= 0.19.1
5. scipy >= 1.0.0
6. gensim >= 3.3.0
7. matplotlib >= 2.1.1

## File Description
1. general_corpus: contains corpus for general nlp tasks such as stopwords and sentiment dict
2. review: contains review of novels(xlsx files) may be with sentiment labels
3. rf_result: contains result of sentiment analysis
4. misc.py: multiple functions to read file, drop null and flatten list, etc
5. review_preprocess.py: read and preprocess reviews, then generate tf-idf matrix
6. gen_classifier.py: train classifiers and find optimal paras, a randomforest is trained in this case
7. model_analysis.py: analysis the predictive performance of classifier, result saved in rf_result 

## TODO
1. expand the general_corpus, especially the sentiment dict
2. test other classifier
3. use other features exclude tf-idf, doc2vec for example
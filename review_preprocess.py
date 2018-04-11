# -*- coding: utf-8 -*-

import logging
import misc
from gensim import corpora, models, similarities
from scipy.sparse import csr_matrix


def review_import(train_files, test_files):
    '''
    Function to generate the full set

    train_files: list; contains of files paths of train set
    test_files: list; contains of files paths of test set

    return: reviews_all: list; all reviews; label_full: list; all labels; len_train: int; sample size of train set
    '''
    reviews_train, label_train = misc.gen_set(train_files, 'review', 'class')
    len_train = len(reviews_train)

    # import test set
    reviews_test, label_test = misc.gen_set(test_files, 'review', 'class')
    len_test = len(reviews_test)
    logging.info("{} test samples catched".format(len_test))

    # get the full set
    reviews_all = reviews_train + reviews_test
    label_all = label_train + label_test
    len_all = len(reviews_all)
    logging.info("{} all samples catched, {} train, {} test".format(len_all, len_train, len_test))

    return reviews_all, label_all, len_train


def review_clean(reviews_all, label_full, len_train, dict_path):
    '''
    Function to clean origin train set and test set, basically cut the reviews and drop the null segements

    reviews_all: list; contains reviews of train set and test set
    label_full: list; contains labels of train set and test set 
    len_train: int; sample size of train set
    dict_path: str; path of sentiment dict

    return: review_all_not_null: list; all not null reviews
            label_all_not_null: list; all label corresponding to not null reviews
            len_train_not_null: int; sample size of not null train set
    '''
    # import the sentiment dict
    word_list = misc.read_col(dict_path, '词语')
    logging.info("{} sentiment words catched".format(len(word_list)))

    # get the insection of each review in full set and sentiment dict
    review_all_seg = [[word for word in word_list if word in review] for review in reviews_all]
    review_all_not_null, label_all_not_null = misc.drop_null(review_all_seg, label_full)
    len_train_not_null = len([i for i in range(len(review_all_seg[:len_train])) if review_all_seg[i]])
    len_test_not_null = len(review_all_not_null) - len_train_not_null
    logging.info("{} train reviews and {} test reviews after dropping null element".format(len_train_not_null, len_test_not_null))

    return review_all_not_null, label_all_not_null, len_train_not_null


def gen_tfidf(review_list, len_train):
    '''
    Function to generate the design matrix 

    review_list: list; contains all reviews
    len_train: int; sample size of train set

    return: X_train: sparse matrix; design matrix of train set
            X_test: sparse matrix; design matrix of test set
    '''
    # start to compute tf-idf of each word
    logging.info("start generating tfidf matrix")
    dictionary = corpora.Dictionary(review_list)
    corpus = [dictionary.doc2bow(review) for review in review_list]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # generate the design matrix
    # rows = range(len(corpus_tfidf))
    rows = misc.flat_list([[i]*len(line) for i, line in enumerate(corpus_tfidf)])
    cols = misc.flat_list([[ele[0] for ele in line] for line in corpus_tfidf])
    data = misc.flat_list([[ele[1] for ele in line] for line in corpus_tfidf])
    rows = list(rows)
    cols = list(cols)
    data = list(data)
    tfidf_matrix = csr_matrix((data, (rows, cols))).toarray()
    logging.info("{} rows of tfidf matrix catched".format(len(tfidf_matrix[0])))
    # split the design matrix into train and test
    X_train = tfidf_matrix[:len_train]
    X_test = tfidf_matrix[len_train:]

    return X_train, X_test
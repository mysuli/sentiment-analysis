# -*- coding: utf-8 -*-

import pandas as pd

def read_col(filepath, col_name):
    '''
    Function to read one column of xlsx files

    filepath: str; path to xlsx file 
    col_name: str; name of column to be read

    return: list; 
    '''

    return list(pd.read_excel(filepath)[col_name])


def read_review(filepath, review_col, label_col):
    '''
    Function to read review

    filepath: str; path to xlsx file which must contain review and setiment class label
    review_col: str; column name of review
    label_col: str; column name of sentiment class label

    return: list; 2 lists, review list and label list
    '''
    df = pd.read_excel(filepath)
    reviews = [str(review).strip() for review in df[review_col]]
    labels = list(df[label_col])

    return reviews, labels


def gen_set(file_path_list, review_col, label_col):
    '''
    Function to read xlsx files and merge into 1 set

    train_list: list; contain paths of each xlsx file 
    review_col: str; column name of review
    label_col: str; column name of sentiment class label
    '''
    review_full = list()
    label_full = list()
    if not isinstance(file_path_list, list):
        file_path_list = [file_path_list]
    for file_path in file_path_list:
        review_tmp, label_tmp = read_review(file_path, review_col, label_col)
        review_full += review_tmp
        label_full += label_tmp

    return review_full, label_full


def drop_null(review_list, label_list):
    '''
    Function to drop null element in dataset and the labels of corresponding indice

    data_list: list; list of review
    label_list: list; list of label

    return: 2 lists, review list without null review and label list drop corresponding element
    '''
    not_null_idx = [i for i in range(len(review_list)) if review_list[i]]
    review_not_null = [review_list[i] for i in not_null_idx]
    label_not_null = [label_list[i] for i in not_null_idx]

    return review_not_null, label_not_null


def flat_list(nested_list):
    '''
    Function to flat a list with any nesting depth

    nested_list: list; each ele is list, int, str or tuple

    return: list; a flatten list
    '''
    for ele in nested_list:
        if not isinstance(ele, (list, tuple)):
            yield ele
        else:
            yield from flat_list(ele)

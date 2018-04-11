# -*- coding: utf-8 -*- 

import logging
import misc
import review_preprocess
import gen_classifier
import model_analysis

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')


def main():
    # import train set and test set
    train_files = ['./review/rgnstsh_review_copy.xlsx',
                   './review/shrzlzw_review_copy.xlsx']
    test_files = ['./review/lgyx_review_copy.xlsx']
    reviews_all, label_all, len_train = review_preprocess.review_import(train_files, test_files)
    # cut review into sentiment words and filtered the null reviews
    reviews_filtered = review_preprocess.review_clean(reviews_all, label_all, len_train, './general_corpus/词库.xls')
    review_all_not_null = reviews_filtered[0]
    label_all_not_null = reviews_filtered[1]
    len_train_not_null = reviews_filtered[2]
    # generate tfidf matrix of train and test set
    X_train, X_test = review_preprocess.gen_tfidf(review_all_not_null, len_train_not_null)
    # generate the rf model with grid search
    logging.info("start grid search")
    parameters = {'n_estimators': range(10, 100, 10), }
    label_train = label_all_not_null[:len_train_not_null]
    rf = gen_classifier.gen_rf(X_train, label_train, parameters)
    # analysis the performance of rf model
    pred = rf.predict(X_test)
    label_test = label_all_not_null[len_train_not_null:]
    pred_indice = model_analysis.pred_evaluation(pred, label_test, True, "./rf_result")
    for k, v in pred_indice.items():
        logging.info("{} of current model is {}".format(k, v))
    
    return 0


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.info("error {} happen!".format(e))

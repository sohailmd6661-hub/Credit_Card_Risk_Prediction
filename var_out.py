import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import seaborn as sns
import logging
from logging_code import setup_logging
logger = setup_logging("var_out")
from scipy.stats import yeojohnson

def vt_outliers(X_train_num , X_test_num):
    try:
        logger.info(f"Before Train Column Name : {X_train_num.columns}")
        logger.info(f"Before Test Column Name : {X_test_num.columns}")

        for i in X_train_num.columns:
            X_train_num[i+'_yeo'],lam_value = yeojohnson(X_train_num[i])
            X_test_num[i + '_yeo'], lam_value = yeojohnson(X_test_num[i])
            X_train_num = X_train_num.drop([i],axis=1)
            X_test_num = X_test_num.drop([i],axis=1)
            # trimming
            iqr = X_train_num[i+'_yeo'].quantile(0.75) - X_train_num[i+'_yeo'].quantile(0.25)
            upper_limit = X_train_num[i+'_yeo'].quantile(0.75) + (1.5 * iqr)
            lower_limit = X_train_num[i+'_yeo'].quantile(0.25) - (1.5 * iqr)
            X_train_num[i+'_trim'] = np.where(X_train_num[i+'_yeo'] > upper_limit ,upper_limit ,
                                              np.where(X_train_num[i+'_yeo'] < lower_limit , lower_limit,X_train_num[i+'_yeo']))

            X_test_num[i + '_trim'] = np.where(X_test_num[i + '_yeo'] > upper_limit, upper_limit,
                                                np.where(X_test_num[i + '_yeo'] < lower_limit, lower_limit,
                                                         X_test_num[i + '_yeo']))

            X_train_num = X_train_num.drop([i+'_yeo'],axis=1)
            X_test_num = X_test_num.drop([i+'_yeo'],axis=1)

        logger.info(f"After Train Column Name : {X_train_num.columns}")
        logger.info(f"AFter Test Column Name : {X_test_num.columns}")

        return X_train_num , X_test_num
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")
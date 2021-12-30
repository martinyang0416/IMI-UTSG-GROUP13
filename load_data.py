import pandas as pd
from scipy import stats
import numpy as np


def load_data(file):
    data = pd.read_csv(file)
    return data


def get_labels(labels):
    labels_array = labels[["Column"]].values
    labels_list = []
    for labels in labels_array:
        labels_list.append(labels[0])
    labels_list.remove('Period_Numeric')
    labels_list.remove('CUSTOMER_ID (Tokenized)')
    labels_list.remove('STATEMENT_DATE')
    labels_list.remove('RATING_SAVE_DATE')
    labels_list.remove('Corp_Residence_Country_Code')
    labels_list.remove('BR Code')
    labels_list.remove('Period')
    return labels_list


def get_x_y_labels(labels_list):
    y_easy_labels = labels_list[1:2]
    y_med_labels = labels_list[0:1]
    x_labels = labels_list[2:]
    return np.array(y_easy_labels), np.array(y_med_labels), np.array(x_labels)


def get_X_Y(data_train):
    X_array = data_train.values
    X_list = X_array.tolist()
    X_Final = []
    Y_Easy_Final = []
    Y_Med_Final = []
    X_list_empty = []
    for item in X_list:
        X_list_empty.append(item[5:])
    for item in X_list_empty:
        X_Final.append(item[2:])
        Y_Easy_Final.append(item[1:2])
        Y_Med_Final.append(item[0:1])
    return np.array(X_Final), np.array(Y_Easy_Final), np.array(Y_Med_Final)


def get_no_na(data_train):
    X_array = data_train.values
    X_list = X_array.tolist()
    X_no_na = []
    Y_no_na_Easy = []
    Y_no_na_Med = []
    X_list_empty = []
    for item in X_list:
        X_list_empty.append(item[5:])
    for item in X_list_empty:
        new_list = []
        for items in item:
            if not np.isnan(items):
                new_list.append(items)
        if len(new_list) == len(item):
            X_no_na.append(item[2:])
            Y_no_na_Easy.append(item[1:2])
            Y_no_na_Med.append(item[0:1])
    return np.array(X_no_na), np.array(Y_no_na_Easy), np.array(Y_no_na_Med)


if __name__ == "__main__":
    data_train = load_data('cmeg_df_case_competition_scrambled_train.csv')
    data_val_test = load_data('general_industries_df_case_competition_scrambled_train.csv')
    labels = load_data('cmeg_df_case_competition_scrambled_train_datakey.csv')
    labels_val_test = load_data('general_industries_df_case_competition_scrambled_train_datakey.csv')

    labels_list = get_labels(labels)
    for items in labels_list:
        sta, p = stats.normaltest(data_train[[items]].values, nan_policy='omit')
        for item in p:
            if item > 0.05:
                print(items)

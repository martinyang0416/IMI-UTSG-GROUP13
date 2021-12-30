from sklearn import ensemble
from sklearn.model_selection import cross_val_score
import load_data
import numpy as np


np.random.seed(0)

if __name__ == "__main__":
    data_train = load_data.load_data('cmeg_df_case_competition_scrambled_train.csv')
    data_val_test = load_data.load_data('general_industries_df_case_competition_scrambled_train.csv')
    labels = load_data.load_data('cmeg_df_case_competition_scrambled_train_datakey.csv')
    labels_val_test = load_data.load_data('general_industries_df_case_competition_scrambled_train_datakey.csv')

    X_no_na, Y_no_na_Easy, Y_no_na_Med = load_data.get_no_na(data_train)
    y_easy_labels, y_med_labels, x_labels = load_data.get_x_y_labels(labels)

    X_no_na_test, Y_no_na_Easy_test, Y_no_na_Med_test = load_data.get_no_na(data_val_test)
    new_list = []
    num = 0
    tt = 0
    num_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    for num in num_list:
        clf = ensemble.RandomForestClassifier(n_estimators=num, max_depth=None, min_samples_split=2, random_state=0).\
            fit(X_no_na, Y_no_na_Easy)
        scores = cross_val_score(clf, X_no_na, Y_no_na_Easy)
        print(scores.mean())
        print(num)




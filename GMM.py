import load_data
from sklearn import mixture
import numpy as np

np.random.seed(0)

if __name__ == "__main__":
    data_train = load_data.load_data('cmeg_df_case_competition_scrambled_train.csv')
    data_val_test = load_data.load_data('general_industries_df_case_competition_scrambled_train.csv')
    labels = load_data.load_data('cmeg_df_case_competition_scrambled_train_datakey.csv')
    labels_val_test = load_data.load_data('general_industries_df_case_competition_scrambled_train_datakey.csv')

    X_no_na, Y_no_na_Easy, Y_no_na_Med = load_data.get_no_na(data_train)
    y_easy_labels, y_med_labels, x_labels = load_data.get_x_y_labels(labels)

    clf = mixture.GaussianMixture(n_components=20, covariance_type='full', max_iter=10000, reg_covar=1000)
    clf.fit(X_no_na)
    result = clf.predict(X_no_na)
    test_accuracy = np.mean(result == Y_no_na_Easy)
    print(test_accuracy)
    print(clf.fit(X_no_na))

















from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import load_data
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

if __name__ == "__main__":

    data_train = load_data.load_data('cmeg_df_case_competition_scrambled_train.csv')
    data_val_test = load_data.load_data('general_industries_df_case_competition_scrambled_train.csv')
    labels = load_data.load_data('cmeg_df_case_competition_scrambled_train_datakey.csv')
    labels_val_test = load_data.load_data('general_industries_df_case_competition_scrambled_train_datakey.csv')

    X_no_na, Y_no_na_Easy, Y_no_na_Med = load_data.get_no_na(data_train)
    y_easy_labels, y_med_labels, x_labels = load_data.get_x_y_labels(labels)

    X_no_na_test, Y_no_na_Easy_test, Y_no_na_Med_test = load_data.get_no_na(data_val_test)

    X = X_no_na.tolist()
    X_1 = X_no_na_test.tolist()
    for item in X_1:
        X.append(item)
    X = np.array(X)
    Y = Y_no_na_Easy_test.tolist()
    Y_1 = Y_no_na_Easy.tolist()
    for item in Y_1:
        Y.append(item)
    Y = np.array(Y)

    '''pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    plt.plot([i for i in range(X.shape[1])],
             [np.sum(pca.explained_variance_ratio_[:i + 1]) for i in range(X.shape[1])])
    plt.show()'''
    pca = PCA(n_components=4)
    pca.fit(X)
    # PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,svd_solver='auto', tol=0.0, whiten=False)
    x = pca.transform(X)

    x_train, x_val_test, y_train, y_val_test = train_test_split(x, Y, test_size=0.4)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.4)

    '''for num in range(10, 51):
        clf = ensemble.RandomForestClassifier(n_estimators=num, max_depth=None, min_samples_split=2, random_state=0).\
            fit(x_train_reduction, np.ravel(Y_no_na_Easy, order='C'))
        scores = cross_val_score(clf, x_train_reduction,  np.ravel(Y_no_na_Easy, order='C'))
        new_list_1.append(scores.mean())
        new_list_2.append(num)
    plt.plot(new_list_1, new_list_2)
    plt.show()'''
    list_1 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for num in list_1:
        bdt = AdaBoostClassifier(ensemble.RandomForestClassifier(n_estimators=num, max_depth=40, min_samples_split=4),
                                 algorithm="SAMME.R",
                                 n_estimators=60, learning_rate=0.005)  # num = 3 最好 Score: 99.77621218400331 Val
        # score: 78.30501450476585 Test score: 78.8688626476072
        bdt.fit(x_train, np.ravel(y_train, order='C'))
        train_prediction = bdt.predict(x_train)
        val_prediction = bdt.predict(x_val)
        test_prediction = bdt.predict(x_test)
        print(num)
        print("Score:", accuracy_score(train_prediction, np.ravel(y_train, order='C')) * 100)
        print("Val score:", accuracy_score(val_prediction, np.ravel(y_val, order='C')) * 100)
        print("Test score:", accuracy_score(test_prediction, np.ravel(y_test, order='C')) * 100)

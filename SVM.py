from sklearn import svm
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pickle
import dataProc


def SVMClassifier(X_train_path, y_train_path, X_test_path, y_test_path):
    X_train = pickle.load(open(X_train_path))
    X_test = pickle.load(open(X_test_path))
    y_train = pickle.load(open(y_train_path))
    y_test = pickle.load(open(y_test_path))

    bestDCgamma = []
    cost = 0.0
    cost_list = []
    fp = './output/SVM_cst3.txt'
    f = open(fp, 'w')
    f.write('d\tC\tgamma\taccu\n')
    for d in range(1, 4):
        poly = PolynomialFeatures(d)
        X_train_poly = poly.fit_transform(X_train)
        scaled_X_train, scaler = dataProc.standardize_X(X_train_poly)
        X_test_poly = poly.fit_transform(X_test)
        scaled_X_test = scaler.transform(X_test_poly)


        C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        gamma_list = [30, 100, 300]

        for c in C_list:
            for gamma in gamma_list:
                clf = svm.SVC(C=c, gamma=gamma)

                clf.fit(scaled_X_train, y_train)

                prdct_y = clf.predict(scaled_X_test)

                y_test = np.array(y_test)

                sample_num = 0
                accu = 0.0

                for index, i in enumerate(y_test):
                    sample_num += 1
                    if y_test[index] == prdct_y[index]:
                        accu += 1

                accu = accu/sample_num
                cost_list.append([d, c, gamma, accu])

                print("(d= {0}, C = {1}, gamma = {2}) is {3}").format(d, c, gamma, accu)
                f.write(str(d) + '\t' + str(c) + '\t' + str(gamma) + '\t' + str(accu) + '\n')
                if accu > cost:
                    bestDCgamma = [d, c, gamma]
                    cost = accu

    print(bestDCgamma, cost)
    f.flush()
    f.close()


def SVMClassifier2(X_train_path, y_train_path, X_test_path, y_test_path):
    X_train = pickle.load(open(X_train_path))
    X_test = pickle.load(open(X_test_path))
    y_train = pickle.load(open(y_train_path))
    y_test = pickle.load(open(y_test_path))

    bestCgamma = []
    cost = 0.0
    cost_list = []

    scaled_X_train, scaler = dataProc.standardize_X(X_train)
    scaled_X_test = scaler.transform(X_test)

    C_list = [30, 100, 300]

    for c in range(10, 301):
        for gamma in range(1, 100):
            clf = svm.SVC(C=c, gamma=gamma)

            clf.fit(scaled_X_train, y_train)

            prdct_y = clf.predict(scaled_X_test)

            y_test = np.array(y_test)

            sample_num = 0
            accu = 0.0

            for index, i in enumerate(y_test):
                sample_num += 1
                if y_test[index] == prdct_y[index]:
                    accu += 1

            accu = accu/sample_num
            cost_list.append([c, gamma, accu])

            print("( C = {0}, gamma = {1}) is {2}").format(c, gamma, accu)
            if accu > cost:
                bestCgamma = [c, gamma]
                cost = accu

    pickle.dump(cost_list, open('cst_SVM.p', 'wb'))
    print(bestCgamma, cost)


X_train_path = 'X_train.p'
y_train_path = 'y_train.p'
X_test_path = 'X_test.p'
y_test_path = 'y_test.p'

SVMClassifier(X_train_path, y_train_path, X_test_path, y_test_path)
"""
SVMClassifier2(X_train_path, y_train_path, X_test_path, y_test_path)
"""
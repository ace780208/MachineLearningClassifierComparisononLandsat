from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pickle
import dataProc


def ANNClassifier(X_train, y_train, X_test, y_test):

    bestLyrAlphaIter = []
    cost = 0.0
    cost_list = []
    fp = './output/ANN_cst_d3_test.txt'
    f = open(fp, 'w')
    f.write('d\tlyr\talpha\titer\trate\taccu\n')
    for d in range(3, 4):
        poly = PolynomialFeatures(d)
        X_train_poly = poly.fit_transform(X_train)
        scaled_X_train, scaler = dataProc.standardize_X(X_train_poly)
        X_test_poly = poly.fit_transform(X_test)
        scaled_X_test = scaler.transform(X_test_poly)

        hid_layer = [3]
        Alpha_list = [1, 3, 10]
        iter_list = [3000, 10000, 30000]
        learn_rate = [0.001, 0.003]

        for l in hid_layer:
            feature_num = scaled_X_train.shape[1]
            inLyr = [feature_num] * l
            inLyr = tuple(inLyr)
            for Alpha in range(1, 11, 2):
                for iter in range(3000, 30000, 3000):
                    for rate in learn_rate:
                        clf = MLPClassifier(hidden_layer_sizes=inLyr,
                                             alpha=Alpha, learning_rate_init=rate, max_iter=iter)

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
                        cost_list.append([d, l, Alpha, iter, rate, accu])

                        print("(d= {0}, layer = {1}, Alpha = {2}, iter = {3}, rate = {4}) is {5}"
                              .format(d, l, Alpha, iter, rate, accu))
                        f.write(str(d) + '    ' + str(l) + '  ' + str(Alpha) + '   ' + str(iter) + '  ' + str(rate)
                                + '   ' + str(accu) + '\n')
                        if accu > cost:
                            bestLyrAlphaIter = [d, l, Alpha, iter, rate]
                            cost = accu
    print(bestLyrAlphaIter, cost)
    f.flush()
    f.close()


filepath = './mydata/training_data2.txt'
X, y = dataProc.readdata(filepath, "GRID_CODE")

from sklearn.model_selection import train_test_split
X_use, X_rest, y_use, y_rest = train_test_split(X, y, test_size=0.8, random_state=42)
del X_rest, y_rest
X_train, X_test, y_train, y_test = dataProc.split_data(X_use, y_use)

ANNClassifier(X_train, y_train, X_test, y_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
import pickle
import numpy as np
import dataProc


def DTClassifier(X_train, y_train, X_test, y_test):
    train_sample_size = len(X_train)

    cost_list = []
    cost = 0.0
    bestDLeaf = []
    fp = './output/DT_cst3.txt'
    f = open(fp, 'w')
    f.write('d\tleaf\taccu\n')
    for d in range(1, 2):
        poly = PolynomialFeatures(d)
        X_train_poly = poly.fit_transform(X_train)
        scaled_X_train, scaler = dataProc.standardize_X(X_train_poly)
        X_test_poly = poly.fit_transform(X_test)
        scaled_X_test = scaler.transform(X_test_poly)

        # the following performs the pre-pruning approach
        minSampleLeaf = [x for x in range(1, 70, 1)]

        for leaf in minSampleLeaf:
            clf = DecisionTreeClassifier(min_samples_leaf=leaf)
            clf = clf.fit(scaled_X_train, y_train)
            y_predict = clf.predict(scaled_X_test)

            y_test = np.array(y_test)

            sample_num = 0
            accu = 0.0

            for index, i in enumerate(y_test):
                sample_num += 1
                if y_test[index] == y_predict[index]:
                    accu += 1

            accu = accu/sample_num
            cost_list.append([d, leaf, accu])

            print("(d = {0}, leaf = {1}) is {2}".format(d, leaf, accu))
            f.write(str(d)+'\t'+str(leaf)+'\t'+str(accu)+'\n')
            if accu > cost:
                bestDLeaf = [d, leaf]
                cost = accu

    print(bestDLeaf, cost)
    f.flush()
    f.close()

filepath = './mydata/training_data2.txt'
X, y = dataProc.readdata(filepath, "GRID_CODE")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = dataProc.split_data(X, y)

#DTClassifier(X_train, y_train, X_test, y_test)

print (len(y_train))
print (9./len(y)*100)
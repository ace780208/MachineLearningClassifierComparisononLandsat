from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import pickle
import numpy as np
import dataProc


def RFClassifier(X_train, y_train, X_test, y_test):
    num_feat = len(X_train[0])
    num_tree = [x for x in range(10, 1010, 10)]

    cost = 0.0
    bestDFeatTrees = []
    fp = './output/RF_test.txt'
    f = open(fp, 'w')
    f.write('d\tfeatures\ttrees\taccu\n')
    for d in range(3, 4):
        poly = PolynomialFeatures(d)
        X_train_poly = poly.fit_transform(X_train)
        scaled_X_train, scaler = dataProc.standardize_X(X_train_poly)
        X_test_poly = poly.fit_transform(X_test)
        scaled_X_test = scaler.transform(X_test_poly)

        for feat in range(1, num_feat+1):
            for tree in num_tree:
                clf = RandomForestClassifier(n_estimators=tree, max_features=feat)
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

                print("(d = {0}, features = {1}, trees = {2} is {3})".format(d, feat, tree, accu))
                f.write(str(d)+'\t'+str(feat)+'\t'+str(tree)+'\t'+str(accu)+'\n')
                if accu > cost:
                    bestDFeatTrees = [d, feat, tree]
                    cost = accu

    print(bestDFeatTrees, cost)
    f.flush()
    f.close()

'''
filepath = './mydata/FNNR/FNNR_trainingdata.txt'
X, y = dataProc.readdata(filepath, "GRID_CODE")

X_train, X_test, y_train, y_test = dataProc.split_data(X, y)
'''

X_train = pickle.load(open('X_train.p'))
X_test = pickle.load(open('X_test.p'))
y_train = pickle.load(open('y_train.p'))
y_test = pickle.load(open('y_test.p'))
RFClassifier(X_train, y_train, X_test, y_test)
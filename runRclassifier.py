import pandas as pd
import pyper
from sklearn.preprocessing import PolynomialFeatures
import dataProc
import numpy as np
import psycopg2
import psycopg2.errorcodes as errorcodes
from datetime import datetime

def runtreeclassifier(polyD):
    # connect to database to load training and image
    starttime = datetime.now()
    DB_NAME = 'MLCdata'
    USER_NAME = 'Shih_admin'
    PASSWD = 'racoon790713'
    try:
        connection = psycopg2.connect(database=DB_NAME, user=USER_NAME, password=PASSWD)
        print("connection to '%s' succes!"%(DB_NAME))

    except Exception as e:
        print("connection to '%s' failed"%(DB_NAME))
        print(e)
        print(errorcodes.lookup(e.pgcode))

    # load training data from database
    query = 'Select * From accratraining Order by pointid;'
    cursor = connection.cursor()
    cursor.execute(query)
    traindata = cursor.fetchall()
    traindata = np.array(list(traindata))

    if polyD > 1:
        trainX = traindata[:, 3:]
        poly = PolynomialFeatures(polyD)
        trainX = poly.fit_transform(trainX)
        traindata = np.concatenate((traindata[:, :3], trainX[:, 1:]), axis=1)
    train_pddf = pd.DataFrame(traindata)
    colname = ['objectid', 'pointid', 'class']
    dtype = [int, int, str]
    for i in range(len(train_pddf.columns)-3):
        colname.append('v'+str(i+1))
        dtype.append(float)
    train_pddf.columns = colname
    for i in range(len(train_pddf.columns)):
        train_pddf[colname[i]] = train_pddf[colname[i]].astype(dtype[i])

    # set up tree classifier and train it in R
    r = pyper.R(RCMD=r'C:\Program Files\R\R-3.3.3\bin\R.exe', use_pandas='True')
    r("setwd('E:/2016_MachineLearningClassifier/mydata')")
    r('require(ISLR)')
    r('require(tree)')
    # feed training data into R
    r.assign("df1", train_pddf)
    # Warning!! must get the name of column in order to get the prediction work
    r('set.seed(1011)')
    r('train = sample(1:nrow(df1),as.integer(nrow(df1)*0.8))')
    r('tree.landcover = tree(class~.-objectid-pointid,data=df1, subset=train)')
    r('cv.landcover=cv.tree(tree.landcover,FUN=prune.misclass)')
    r('prune.landcover=prune.misclass(tree.landcover,best=9)')

    # connect to image database below and query for the total number of pixels
    query = 'Select Count(*) From accra8bands;'
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    pixelsize = result[0]
    del result

    # create a csv file for store all classified pixels
    outfp = './output/accra_R_treepx.csv'
    import csv
    f = open(outfp, 'w')
    wr = csv.writer(f, delimiter=',')
    header = ['pointid', 'class']
    wr.writerow(header)
    # break all the pixels into multiple sets for classifier predicting
    loop = int(pixelsize/10000)
    if (pixelsize % 10000) != 0:
        loop = loop+1
    print('start to query...')
    for i in range(loop):
        query = 'Select * From accra8bands Where pointid>'+str(i*10000) + \
                ' and pointid<= ' + str((i+1)*10000) + ' Order by pointid;'
        cursor.execute(query)
        pixel10k = cursor.fetchall()

        # get the spectra info from database and make it as data frame with pandas
        pixel10k = np.array(list(pixel10k))
        if polyD > 1:
            X = pixel10k[:, 3:]
            poly = PolynomialFeatures(polyD)
            X = poly.fit_transform(X)
            pixel10k = np.concatenate((pixel10k[:, :3], X[:, 1:]), axis=1)
        pixel_pddf = pd.DataFrame(pixel10k)
        pixel_pddf.columns = colname
        for i in range(len(pixel_pddf.columns)):
            pixel_pddf[colname[i]] = pixel_pddf[colname[i]].astype(dtype[i])

        r.assign('df_pred', pixel_pddf)

        print('start to predict...')
        r('tree.pred=predict(prune.landcover,df_pred, type="class")')
        # done work, send R result back to python

        predclass = r.get('tree.pred')
        predclass = predclass.astype(int)
        predclass = predclass.reshape(predclass.shape[0], -1)
        id = pixel10k[:,1]
        id = id.astype(int)
        id = id.reshape(id.shape[0], -1)
        pixelout = np.concatenate((id, predclass), axis=1)
        print(pixelout)
        print(pixelout.shape)
        wr.writerows(pixelout)
    f.flush()
    f.close()
    connection.commit()
    cursor.close()
    connection.close()
    endtime = datetime.now()
    dur = endtime - starttime
    print("time usage: " + str(dur.seconds) + " seconds")


def runRFclassifier(polyD):
    starttime = datetime.now()
    # connect to database to load training and image
    DB_NAME = 'MLCdata'
    USER_NAME = 'Shih_admin'
    PASSWD = 'racoon790713'
    try:
        connection = psycopg2.connect(database=DB_NAME, user=USER_NAME, password=PASSWD)
        print("connection to '%s' succes!" % (DB_NAME))

    except Exception as e:
        print("connection to '%s' failed" % (DB_NAME))
        print(e)
        print(errorcodes.lookup(e.pgcode))

    # load training data from database
    query = 'Select * From accratraining Order by pointid;'
    cursor = connection.cursor()
    cursor.execute(query)
    traindata = cursor.fetchall()
    traindata = np.array(list(traindata))

    if polyD > 1:
        trainX = traindata[:, 3:]
        poly = PolynomialFeatures(polyD)
        trainX = poly.fit_transform(trainX)
        traindata = np.concatenate((traindata[:, :3], trainX[:, 1:]), axis=1)
    train_pddf = pd.DataFrame(traindata)
    colname = ['objectid', 'pointid', 'class']
    dtype = [int, int, str]
    for i in range(len(train_pddf.columns) - 3):
        colname.append('v' + str(i + 1))
        dtype.append(float)
    train_pddf.columns = colname
    for i in range(len(train_pddf.columns)):
        train_pddf[colname[i]] = train_pddf[colname[i]].astype(dtype[i])

    # set up tree classifier and train it in R
    r = pyper.R(RCMD=r'C:\Program Files\R\R-3.3.3\bin\R.exe', use_pandas='True')
    r("setwd('E:/2016_MachineLearningClassifier/mydata')")
    r.assign("df1", train_pddf)
    r('require(randomForest)')
    r('rf = randomForest(class~.-objectid-pointid, data=df1, mtry=5, ntree=180)')


    # connect to image database below and query for the total number of pixels
    query = 'Select Count(*) From accra8bands;'
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    pixelsize = result[0]
    del result

    # create a csv file for store all classified pixels
    outfp = './output/accra_R_RFpx.csv'
    import csv

    f = open(outfp, 'w')
    wr = csv.writer(f, delimiter=',')
    header = ['pointid', 'class']
    wr.writerow(header)
    # break all the pixels into multiple sets for classifier predicting
    loop = int(pixelsize / 10000)
    if (pixelsize % 10000) != 0:
        loop = loop + 1
    print('start to query...')
    for i in range(loop):
        query = 'Select * From accra8bands Where pointid>' + str(i * 10000) + \
                ' and pointid<= ' + str((i + 1) * 10000) + ' Order by pointid;'
        cursor.execute(query)
        pixel10k = cursor.fetchall()

        # get the spectra info from database and make it as data frame with pandas
        pixel10k = np.array(list(pixel10k))
        if polyD > 1:
            X = pixel10k[:, 3:]
            poly = PolynomialFeatures(polyD)
            X = poly.fit_transform(X)
            pixel10k = np.concatenate((pixel10k[:, :3], X[:, 1:]), axis=1)
        pixel_pddf = pd.DataFrame(pixel10k)
        pixel_pddf.columns = colname
        for j in range(len(pixel_pddf.columns)):
            pixel_pddf[colname[j]] = pixel_pddf[colname[j]].astype(dtype[j])

        r.assign('df_pred', pixel_pddf)
        print('start to predict...')
        print(i)
        r('pred=predict(rf, df_pred)')
        # done work, send R result back to python

        predclass = r.get('pred')
        print(predclass)
        predclass = predclass.astype(int)
        predclass = predclass.reshape(predclass.shape[0], -1)
        id = pixel10k[:, 1]
        id = id.astype(int)
        id = id.reshape(id.shape[0], -1)
        pixelout = np.concatenate((id, predclass), axis=1)
        wr.writerows(pixelout)
    f.flush()
    f.close()
    connection.commit()
    cursor.close()
    connection.close()
    endtime = datetime.now()
    dur = endtime - starttime
    print("time usage: " + str(dur.seconds) + " seconds")


def runSVMclassifier(polyD):
    starttime = datetime.now()
    # connect to database to load training and image
    DB_NAME = 'MLCdata'
    USER_NAME = 'Shih_admin'
    PASSWD = 'racoon790713'
    try:
        connection = psycopg2.connect(database=DB_NAME, user=USER_NAME, password=PASSWD)
        print("connection to '%s' succes!" % (DB_NAME))

    except Exception as e:
        print("connection to '%s' failed" % (DB_NAME))
        print(e)
        print(errorcodes.lookup(e.pgcode))

    # load training data from database
    query = 'Select * From accratraining Order by pointid;'
    cursor = connection.cursor()
    cursor.execute(query)
    traindata = cursor.fetchall()
    traindata = np.array(list(traindata))

    trainX = traindata[:, 3:]
    poly = PolynomialFeatures(polyD)
    trainX = poly.fit_transform(trainX)
    # SVM needs one more step for data preprocessing i.e. feature scaling
    trainX, scaler = dataProc.standardize_X(trainX)
    traindata = np.concatenate((traindata[:, :3], trainX[:, 1:]), axis=1)
    train_pddf = pd.DataFrame(traindata)
    del trainX, traindata
    colname = ['objectid', 'pointid', 'class']
    dtype = [int, int, str]
    for i in range(len(train_pddf.columns) - 3):
        colname.append('v' + str(i + 1))
        dtype.append(float)
    train_pddf.columns = colname
    for i in range(len(train_pddf.columns)):
        train_pddf[colname[i]] = train_pddf[colname[i]].astype(dtype[i])

    # set up SVM classifier and train it in R
    r = pyper.R(RCMD=r'C:\Program Files\R\R-3.3.3\bin\R.exe', use_pandas='True')
    r("setwd('E:/2016_MachineLearningClassifier/mydata')")
    r.assign("df1", train_pddf)
    r('require(e1071)')
    print('starting training...')
    starttime1 = datetime.now()
    r("svmfit = svm(class~.-objectid-pointid, data=df1, kernel='radial', gamma=58, cost=16)")
    endtime1 = datetime.now()
    dur = endtime1 - starttime1
    print("time usage: " + str(dur.days) + " days, " + str(dur.seconds) + " seconds, " + str(
        dur.microseconds) + " musecs.")

    # connect to image database below and query for the total number of pixels
    query = 'Select Count(*) From accra8bands;'
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    pixelsize = result[0]
    del result

    # create a csv file for store all classified pixels
    outfp = './output/accra_R_SVMpx.csv'
    import csv

    f = open(outfp, 'w')
    wr = csv.writer(f, delimiter=',')
    header = ['pointid', 'class']
    wr.writerow(header)
    # break all the pixels into multiple sets for classifier predicting
    loop = int(pixelsize / 10000)
    if (pixelsize % 10000) != 0:
        loop = loop + 1
    print('start to query...')
    for i in range(loop):
        query = 'Select * From accra8bands Where pointid>' + str(i * 10000) + \
                ' and pointid<= ' + str((i + 1) * 10000) + ' Order by pointid;'
        cursor.execute(query)
        pixel10k = cursor.fetchall()

        # get the spectra info from database and make it as data frame with pandas
        pixel10k = np.array(list(pixel10k))
        X = pixel10k[:, 3:]
        X = poly.fit_transform(X)
        X = scaler.transform(X)
        pixel10k = np.concatenate((pixel10k[:, :3], X[:, 1:]), axis=1)
        pixel_pddf = pd.DataFrame(pixel10k)
        del X
        pixel_pddf.columns = colname
        for j in range(len(pixel_pddf.columns)):
            pixel_pddf[colname[j]] = pixel_pddf[colname[j]].astype(dtype[j])

        r.assign('df_pred', pixel_pddf)
        print('start to predict...')
        print(i)
        r('pred=predict(svmfit, df_pred)')
        # done work, send R result back to python

        predclass = r.get('pred')
        predclass = predclass.astype(int)
        predclass = predclass.reshape(predclass.shape[0], -1)
        id = pixel10k[:, 1]
        id = id.astype(int)
        id = id.reshape(id.shape[0], -1)
        pixelout = np.concatenate((id, predclass), axis=1)
        wr.writerows(pixelout)
    f.flush()
    f.close()
    connection.commit()
    cursor.close()
    connection.close()
    endtime = datetime.now()
    dur = endtime - starttime
    print("time usage: " + str(dur.days) + " days, " + str(dur.seconds) + " seconds, " + str(dur.microseconds) + " musecs.")


def runANNclassifier(polyD):
    starttime = datetime.now()
    # connect to database to load training and image
    DB_NAME = 'MLCdata'
    USER_NAME = 'Shih_admin'
    PASSWD = 'racoon790713'
    try:
        connection = psycopg2.connect(database=DB_NAME, user=USER_NAME, password=PASSWD)
        print("connection to '%s' succes!" % (DB_NAME))

    except Exception as e:
        print("connection to '%s' failed" % (DB_NAME))
        print(e)
        print(errorcodes.lookup(e.pgcode))

    # load training data from database
    query = 'Select * From accratraining Order by Random() limit 80000;'
    cursor = connection.cursor()
    cursor.execute(query)
    traindata = cursor.fetchall()
    traindata = np.array(list(traindata))
    traindata = traindata[:, 2:]

    trainX = traindata[:, 1:]
    poly = PolynomialFeatures(polyD)
    trainX = poly.fit_transform(trainX)
    # ANN needs one more step for data preprocessing i.e. feature scaling
    trainX, scaler = dataProc.standardize_X(trainX)
    xlen = len(trainX[0])-1

    # y (class) need to be broken into multiple columns. The number of new columns corresponds to the number of classes
    y = np.unique(traindata[:, 0])
    yout = []
    ytitle = []
    ytype = []
    for yele in y:
        ytitle.append('cls'+yele)
        ytype.append(int)
        cls = traindata[:, 0] == yele
        cls = cls.astype(int)
        cls = cls.tolist()
        yout.append(cls)
    yout = np.array(yout)
    yout = np.transpose(yout)
    traindata = np.concatenate((yout, trainX[:, 1:]), axis=1)
    train_pddf = pd.DataFrame(traindata)
    del trainX, traindata, yout

    # need titles for all independent variables
    colname = ytitle
    dtype = ytype
    xtitle = []
    xtype = []
    for i in range(xlen):
        xtitle.append('v' + str(i + 1))
        xtype.append(float)
    colname = colname + xtitle
    dtype = dtype + xtype

    train_pddf.columns = colname
    for i in range(len(train_pddf.columns)):
        train_pddf[colname[i]] = train_pddf[colname[i]].astype(dtype[i])

    # set up ANN classifier and train it in R
    r = pyper.R(RCMD=r'C:\Program Files\R\R-3.3.3\bin\R.exe', use_pandas='True')
    r("setwd('E:/2016_MachineLearningClassifier/mydata')")
    r.assign("df1", train_pddf)
    r('require(neuralnet)')
    starttime1 = datetime.now()
    # for ANN, create formula is necessary in neuralnet package
    formula = 'formu = as.formula("'
    for i, val in enumerate(ytitle):
        if i == 0:
            formula = formula + val
        else:
            formula = formula+'+'+val
    formula = formula + '~'
    for i, val in enumerate(xtitle):
        if i == 0:
            formula = formula + val
        else:
            formula = formula+'+'+val
    formula = formula + '")'
    print(formula)
    r(formula)
    print('starting training...')
    r("ann = nueralnet(formu, df1, hidden=c(8,8,8), learningrate= 0.001, stepmax= 21000, lifesign='full')")
    print(r('ann'))
    print(r('summary(ann)'))
    endtime1 = datetime.now()
    dur = endtime1 - starttime1
    print("time usage: " + str(dur.days) + " days, " + str(dur.seconds) + " seconds, " + str(
        dur.microseconds) + " musecs.")

    # connect to image database below and query for the total number of pixels
    query = 'Select Count(*) From accra8bands;'
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    pixelsize = result[0]
    del result

    # create a csv file for store all classified pixels
    outfp = './output/accra_R_ANNpx.csv'
    import csv

    f = open(outfp, 'w')
    wr = csv.writer(f, delimiter=',')
    header = ['pointid', 'class']
    wr.writerow(header)
    # break all the pixels into multiple sets for classifier predicting
    loop = int(pixelsize / 10000)
    if (pixelsize % 10000) != 0:
        loop = loop + 1
    print('start to query...')

    rscript = "colnames(pred.result) = c("
    for k, val in enumerate(ytitle):
        if k == 0:
            rscript = rscript + "'" + val + "'"
        else:
            rscript = rscript + ", '" + val + "'"
    rscript = rscript + ")"
    print(rscript)
    for i in range(loop):
        query = 'Select * From accra8bands Where pointid>' + str(i * 10000) + \
                ' and pointid<= ' + str((i + 1) * 10000) + ' Order by pointid;'
        cursor.execute(query)
        pixel10k = cursor.fetchall()

        # get the spectra info from database and make it as data frame with pandas
        pixel10k = np.array(list(pixel10k))
        X = pixel10k[:, 3:]
        X = poly.fit_transform(X)
        X = scaler.transform(X)

        # should process for y??????
        X = X[:, 1:]
        pixel_pddf = pd.DataFrame(X)
        pixel_pddf.columns = xtitle
        for j in range(len(pixel_pddf.columns)):
            pixel_pddf[xtitle[j]] = pixel_pddf[xtitle[j]].astype(xtype[j])

        r.assign('df_pred', pixel_pddf)

        print('start to predict...')
        print(i)
        r('pred=compute(ann, df_pred)')
        r('pred.result = as.data.frame(pred$net.result)')
        r(rscript)
        r('pred.result$class = substr(colnames(pred.result)[max.col(pred.result,ties.method="first")],4,4)')
        r('output = pred.result$class')

        # done work, send R result back to python
        # Warning!! Not getting anything back from R yet, need to fix
        predclass = r.get('output')
        predclass = predclass.astype(int)
        predclass = predclass.reshape(predclass.shape[0], -1)
        id = pixel10k[:, 1]
        id = id.astype(int)
        id = id.reshape(id.shape[0], -1)
        pixelout = np.concatenate((id, predclass), axis=1)
        wr.writerows(pixelout)
    f.flush()
    f.close()
    connection.commit()
    cursor.close()
    connection.close()
    endtime = datetime.now()
    dur = endtime - starttime
    print("time usage: " + str(dur.days) + " days, " + str(dur.seconds) + " seconds, " + str(dur.microseconds) + " musecs.")


runANNclassifier(3)

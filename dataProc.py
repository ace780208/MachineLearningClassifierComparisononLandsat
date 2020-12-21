def readdata(filepath, yfield):
    f = open(filepath, 'r')
    str = f.readline()
    attribute = str.split(',')
    yindx = attribute.index(yfield)
    y = []
    X = []
    str = f.readline()

    while str:
        attribute_num = str.split(',')
        y_tmp = int(float(attribute_num[yindx]))
        y.append(y_tmp)
        x_tmp = attribute_num[yindx+1:]
        x_tmp = [float(int(float(i))) for i in x_tmp]
        X.append(x_tmp)
        str = f.readline()
    return X, y


def standardize_X(X):
    # this function will convert the feature X in to Z score
    import sklearn.preprocessing as pre
    scaler = pre.MinMaxScaler()
    X_minmax = scaler.fit_transform(X)
    return X_minmax, scaler


def split_data(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_data(X, y, list1):
    import matplotlib.pyplot as plt
    stdz_X, avg_X, std_X = standardize_X(X)
    x1 = stdz_X[:, list1[0]]
    x2 = stdz_X[:, list1[1]]
    plt.scatter(x1, x2, s=75, c=y)
    plt.show()


def countClassNum():
    import pickle
    ylist = pickle.load(open('y_train.p'))
    sety = list(set(ylist))
    sety.sort()
    num = []
    for i in sety:
        tmpnum = 0
        for j in ylist:
            if j == i:
                tmpnum += 1

        num.append(tmpnum)
    return sety, num

def exportData(xpicklefile, ypicklefile):
    import pickle
    xlist = pickle.load(open(xpicklefile))
    ylist = pickle.load(open(ypicklefile))
    sety = list(set(ylist))
    sety.sort()
    num = []
    for i in sety:
        tmpnum = 0
        for j in ylist:
            if j == i:
                tmpnum += 1

        num.append(tmpnum)
    return sety, num


def polynomialLayer(intifImg, d, outtifImg):
    from sklearn.preprocessing import PolynomialFeatures
    from osgeo import gdal
    from osgeo import gdalconst
    import sys

    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    ImgDs = gdal.Open(intifImg, gdalconst.GA_ReadOnly)
    if ImgDs is None:
        print('Error Could not read')
        sys.exit()

    cols =ImgDs.RasterXSize
    rows = ImgDs.RasterYSize
    bands = ImgDs.RasterCount
    projection = ImgDs.GetProjection()
    datatype = gdal.GetDataTypeName(ImgDs.GetRasterBand(1).DataType)
    nodataval = ImgDs.GetRasterBand(1).GetNoDataValue()
    geotransform = ImgDs.GetGeoTransform()

    import numpy as np

    poly = PolynomialFeatures(d)

    sampleFeat = []
    mask = ImgDs.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
    for i in range(bands):
        band = ImgDs.GetRasterBand(i+1)
        data = band.ReadAsArray(int(cols/2), int(rows/2), 1, 1)
        value = data[0][0]
        sampleFeat.append(value)
        band = None
    polyfeat = poly.fit_transform([sampleFeat])
    outbands = len(polyfeat[0])
    polyfeat = None

    ImgDsOut = driver.Create(outtifImg, cols, rows, outbands, gdal.GDT_Float32)

    for l in range(outbands):
        imgInFeat = mask
        for r in range(rows):
            for c in range(cols):
                feat = []
                if imgInFeat[r][c] != nodataval:
                    for i in range(bands):
                        band = ImgDs.GetRasterBand(i + 1)
                        data = band.ReadAsArray(c, r, 1, 1)
                        value = data[0][0]
                        feat.append(value)
                        band = None
                    polyFeat = poly.fit_transform([feat])
                    imgInFeat[r][c] = polyFeat[0][l]

        imgInFeat = np.array(imgInFeat)
        print ('this is img')
        print (imgInFeat)
        out_band = ImgDsOut.GetRasterBand(l+1)
        
        out_band.WriteArray(imgInFeat, 0, 0)
        out_band.SetNoDataValue(nodataval)
        print ('Done layer {0}'.format(l+1))
        out_band = None

    allband = None
    ImgDsOut.SetProjection(projection)
    ImgDsOut.SetGeoTransform(geotransform)
    ImgDsOut = None

    print('Done poly!')


def loadHyperTxt(filepath):
    import numpy as np
    data = np.loadtxt(filepath, skiprows=1)
    x = sorted(set(data[:,0].tolist()))
    y = sorted(set(data[:,1].tolist()))
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(y), len(x)))

    for i in data:
        Xindex = x.index(i[0])
        Yindex = y.index(i[1])
        Z[Yindex][Xindex] = i[2]
    Z = np.around(Z, decimals=5)
    Z = Z.tolist()
    return X, Y, Z


def plotContour(d1, d2, accu):

    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    x = d1
    y = d2
    z = accu
    plt.figure()
    cp = plt.contourf(x, y, z, cmap=cm.gray)
    plt.colorbar(cp)
    plt.title('(d)')
    plt.xlabel('C')
    plt.ylabel('gamma')
    plt.show()

    '''
    import plotly
    plotly.tools.set_credentials_file(username='ace780208', api_key='Qtz6Z2VayGWH9wKYOB6n')
    import plotly.plotly as py
    import plotly.graph_objs as go
    data = [
        go.Contour(
            z=accu,
            x=d1,
            y=d2,
            line=dict(smoothing=0.85),
        )]
    py.iplot(data)
'''


def raster2txt(img, outtxt):
    from osgeo import gdal
    from osgeo import gdalconst
    import sys

    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    ImgDs = gdal.Open(img, gdalconst.GA_ReadOnly)
    if ImgDs is None:
        print('Error Could not read')
        sys.exit()

    cols =ImgDs.RasterXSize
    rows = ImgDs.RasterYSize
    bands = ImgDs.RasterCount
    projection = ImgDs.GetProjection()
    datatype = gdal.GetDataTypeName(ImgDs.GetRasterBand(1).DataType)
    nodataval = ImgDs.GetRasterBand(1).GetNoDataValue()
    geotransform = ImgDs.GetGeoTransform()

    f = open(outtxt, 'w')
    f.write('ncols\t' + str(cols) + '\n')
    f.write('nrows\t' + str(rows) + '\n')
    f.write('xllcorner\t' + str(geotransform[0]) + '\n')
    f.write('yllcorner\t' + str(geotransform[3]) + '\n')
    f.write('cellsize\t' + str(geotransform[1]) + '\n')
    f.write('NODATA_value\t' + str(nodataval) + '\n')

    for i in range(bands):
        if i != bands-1:
            f.write('band' + str(i+1) + '\t')
        else:
            f.write('band' + str(i+1) + '\n')

    for r in range(rows):
        for c in range(cols):
            for i in range(bands):
                band = ImgDs.GetRasterBand(i + 1)
                data = band.ReadAsArray(c, r, 1, 1)
                value = data[0][0]
                if i != bands - 1:
                    f.write(str(value) + '\t')
                else:
                    f.write(str(value) + '\n')
                band = None
        print('done row ' + str(r))
    f.flush()
    f.close()
    ImgDs = None
    print('done img2txt!')

"""
X, Y, Z = loadHyperTxt('./output/FNNR/FNNR_SVM_cst2.txt')
plotContour(X, Y, Z)
"""
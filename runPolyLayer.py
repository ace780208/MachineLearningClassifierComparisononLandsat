import dataProc

inImg = './image/accra2002.tif'
outImg = './mydata/accra_D2.tif'

dataProc.polynomialLayer(inImg, 2, outImg)

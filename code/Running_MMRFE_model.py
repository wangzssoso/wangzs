# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:28:58 2020

@author: wangzs@igsnrr.ac.cn
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import type_of_target
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import explained_variance_score,max_error,mean_squared_error, mean_absolute_error,r2_score
import joblib
nr1=510
nc1=710
hd=['ncols         710',\
    'nrows         510',\
    'xllcorner     70',\
    'yllcorner     10',\
    'cellsize      0.1',\
    'NODATA_value  -9999']

fdir='I:/MsTMIP/MsTMP_China/bomes_plot_data/'
fdir2='I:/MsTMIP/MsTMP_China/code_bioms/'
#fdir='/Volumes/SBP/MsTMIP/MsTMP_China/bomes_plot_data/'
foc=fdir+'plot_2010_biomes_location.dat'
dc=np.loadtxt(foc,skiprows=1)

for year in range (2010,2011):
    fob=fdir+'plot_'+str(year)+'_biomes_Lteration_.dat'
    dx1=np.loadtxt(fob,skiprows=1)
    fndvi=fdir+'china_ndvi_max_'+str(year)+'.asc'
    ndvi=np.loadtxt(fndvi,skiprows=6)
    #print(type_of_target(dx0))
    n=len(dx1[:,0])
    #print(len(dx0[:,2]),len(dx0[:,3:13]))
    xx=np.zeros((n,10))

    xx[:,0:10]=dx1[:,3:13]
    rf2=joblib.load(fdir2+'MMRFE_model.pkl')
    y_pred1=rf2.predict(xx)
    dmpp=np.zeros((nr1,nc1))
    dmpp[:,:]=-9999
    #print(len(y_pred1),type_of_target(y_pred1))
    rsx=np.array(y_pred1)
    #print(type(rsx),rsx.shape,y_pred1.shape)
    for i in range (0,len(rsx)):
        ix=int(dc[i,0])
        ij=int(dc[i,1])
        dmpp[ix,ij]=y_pred1[i]
    for i in range (0,nr1):
        for j in range (0,nc1):

            if dmpp[i,j]<0:
               dmpp[i,j]=-9999
            if ndvi[i,j]<0.1:
               dmpp[i,j]=-9999
    outtif=fdir+'RFS_China_biomesy'+str(year)+'.asc'
    flt=open(outtif,'wb')
    np.savetxt(flt,hd,fmt='%s')
    np.savetxt(flt,dmpp,fmt=['%8.3f']*710,newline='\n')
    flt.close()
    print(year,"over")
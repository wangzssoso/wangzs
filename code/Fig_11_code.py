# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:21:25 2021

@author: wangzs@igsnrr.ac.cn
"""

from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import matplotlib
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
from mpl_toolkits.axisartist.axislines import SubplotZero
from sklearn.metrics import r2_score
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import mpl_toolkits.axisartist.axislines as axislines
import pandas as pd
font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'light',
        'size'   :8,
        }
font1 = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'light',
        'size'   :6,
        }
ty=29
xx=np.zeros((ty))
ldx1=np.zeros((ty,3))
rdx1=np.zeros((ty,3))
ldx2=np.zeros((ty,3))
rdx2=np.zeros((ty,3))
ldx3=np.zeros((ty,3))
rdx3=np.zeros((ty,3))
ldx4=np.zeros((ty,3))
rdx4=np.zeros((ty,3))
ldx5=np.zeros((ty,3))
rdx5=np.zeros((ty,3))
fdir='H:/MsTMIP/MsTMP_China/bomes_plot_data/'
fto=fdir+'china_RF_NDVI_ESM_data.txt'
ff=fdir+'forest_RF_NDVI_ESM_data.dat'
fs=fdir+'shurb_RF_NDVI_ESM_data.dat'
fg=fdir+'grass_RF_NDVI_ESM_data.dat'
fc=fdir+'crop_RF_NDVI_ESM_data.dat'

dx1=np.loadtxt(fto,skiprows=1)
dx2=np.loadtxt(ff,skiprows=1)
dx3=np.loadtxt(fs,skiprows=1)
dx4=np.loadtxt(fg,skiprows=1)
dx5=np.loadtxt(fc,skiprows=1)
xx[:]=dx1[:,0]

ldx1[:,0]=dx1[:,2]/100.0
ldx1[:,1]=dx1[:,5]*100.0
ldx1[:,2]=dx1[:,8]/100.0

ldx2[:,0]=dx2[:,2]/100.0
ldx2[:,1]=dx2[:,5]*100.0
ldx2[:,2]=dx2[:,8]/100.0

ldx3[:,0]=dx3[:,2]/100.0
ldx3[:,1]=dx3[:,5]*100.0
ldx3[:,2]=dx3[:,8]/100.0

ldx4[:,0]=dx4[:,2]/100.0
ldx4[:,1]=dx4[:,5]*100.0
ldx4[:,2]=dx4[:,8]/100.0

ldx5[:,0]=dx5[:,2]/100.0
ldx5[:,1]=dx5[:,5]*100.0
ldx5[:,2]=dx5[:,8]/100.0

rdx1[:,0]=dx1[:,3]/dx1[:,2]
rdx1[:,1]=dx1[:,6]/dx1[:,5]
rdx1[:,2]=dx1[:,9]/dx1[:,8]

rdx2[:,0]=dx2[:,3]/dx2[:,2]
rdx2[:,1]=dx2[:,6]/dx2[:,5]
rdx2[:,2]=dx2[:,9]/dx2[:,8]

rdx3[:,0]=dx3[:,3]/dx3[:,2]
rdx3[:,1]=dx3[:,6]/dx3[:,5]
rdx3[:,2]=dx3[:,9]/dx3[:,8]

rdx4[:,0]=dx4[:,3]/dx4[:,2]
rdx4[:,1]=dx4[:,6]/dx4[:,5]
rdx4[:,2]=dx4[:,9]/dx4[:,8]

rdx5[:,0]=dx5[:,3]/dx5[:,2]
rdx5[:,1]=dx5[:,6]/dx5[:,5]
rdx5[:,2]=dx5[:,9]/dx5[:,8]


ldx1[:,0]=(ldx1[:,0]-np.mean(ldx1[:,0],axis=0))
ldx1[:,1]=(ldx1[:,1]-np.mean(ldx1[:,1],axis=0))
ldx1[:,2]=(ldx1[:,2]-np.mean(ldx1[:,2],axis=0))

ldx2[:,0]=(ldx2[:,0]-np.mean(ldx2[:,0],axis=0))
ldx2[:,1]=(ldx2[:,1]-np.mean(ldx2[:,1],axis=0))
ldx2[:,2]=(ldx2[:,2]-np.mean(ldx2[:,2],axis=0))

ldx3[:,0]=(ldx3[:,0]-np.mean(ldx3[:,0],axis=0))
ldx3[:,1]=(ldx3[:,1]-np.mean(ldx3[:,1],axis=0))
ldx3[:,2]=(ldx3[:,2]-np.mean(ldx3[:,2],axis=0))

ldx4[:,0]=(ldx4[:,0]-np.mean(ldx4[:,0],axis=0))
ldx4[:,1]=(ldx4[:,1]-np.mean(ldx4[:,1],axis=0))
ldx4[:,2]=(ldx4[:,2]-np.mean(ldx4[:,2],axis=0))

ldx5[:,0]=(ldx5[:,0]-np.mean(ldx5[:,0],axis=0))
ldx5[:,1]=(ldx5[:,1]-np.mean(ldx5[:,1],axis=0))
ldx5[:,2]=(ldx5[:,2]-np.mean(ldx5[:,2],axis=0))

yx1_err1=ldx1[:,0].std() * np.sqrt(1/len(ldx1[:,0]) +(ldx1[:,0] - ldx1[:,0].mean())**2 / np.sum((ldx1[:,0] - ldx1[:,0].mean())**2))
yx1_err2=ldx1[:,1].std() * np.sqrt(1/len(ldx1[:,1]) +(ldx1[:,1] - ldx1[:,1].mean())**2 / np.sum((ldx1[:,1] - ldx1[:,1].mean())**2))
yx1_err3=ldx1[:,2].std() * np.sqrt(1/len(ldx1[:,2]) +(ldx1[:,2] - ldx1[:,2].mean())**2 / np.sum((ldx1[:,2] - ldx1[:,2].mean())**2))

yx2_err1=ldx2[:,0].std() * np.sqrt(1/len(ldx2[:,0]) +(ldx2[:,0] - ldx2[:,0].mean())**2 / np.sum((ldx2[:,0] - ldx2[:,0].mean())**2))
yx2_err2=ldx2[:,1].std() * np.sqrt(1/len(ldx2[:,1]) +(ldx2[:,1] - ldx2[:,1].mean())**2 / np.sum((ldx2[:,1] - ldx2[:,1].mean())**2))
yx2_err3=ldx2[:,2].std() * np.sqrt(1/len(ldx2[:,2]) +(ldx2[:,2] - ldx2[:,2].mean())**2 / np.sum((ldx2[:,2] - ldx2[:,2].mean())**2))

yx3_err1=ldx3[:,0].std() * np.sqrt(1/len(ldx3[:,0]) +(ldx3[:,0] - ldx3[:,0].mean())**2 / np.sum((ldx3[:,0] - ldx3[:,0].mean())**2))
yx3_err2=ldx3[:,1].std() * np.sqrt(1/len(ldx3[:,1]) +(ldx3[:,1] - ldx3[:,1].mean())**2 / np.sum((ldx3[:,1] - ldx3[:,1].mean())**2))
yx3_err3=ldx3[:,2].std() * np.sqrt(1/len(ldx3[:,2]) +(ldx3[:,2] - ldx3[:,2].mean())**2 / np.sum((ldx3[:,2] - ldx3[:,2].mean())**2))

yx4_err1=ldx4[:,0].std() * np.sqrt(1/len(ldx4[:,0]) +(ldx4[:,0] - ldx4[:,0].mean())**2 / np.sum((ldx4[:,0] - ldx4[:,0].mean())**2))
yx4_err2=ldx4[:,1].std() * np.sqrt(1/len(ldx4[:,1]) +(ldx4[:,1] - ldx4[:,1].mean())**2 / np.sum((ldx4[:,1] - ldx4[:,1].mean())**2))
yx4_err3=ldx4[:,2].std() * np.sqrt(1/len(ldx4[:,2]) +(ldx4[:,2] - ldx4[:,2].mean())**2 / np.sum((ldx4[:,2] - ldx4[:,2].mean())**2))

yx5_err1=ldx5[:,0].std() * np.sqrt(1/len(ldx5[:,0]) +(ldx5[:,0] - ldx5[:,0].mean())**2 / np.sum((ldx5[:,0] - ldx5[:,0].mean())**2))
yx5_err2=ldx5[:,1].std() * np.sqrt(1/len(ldx5[:,1]) +(ldx5[:,1] - ldx5[:,1].mean())**2 / np.sum((ldx5[:,1] - ldx5[:,1].mean())**2))
yx5_err3=ldx5[:,2].std() * np.sqrt(1/len(ldx5[:,2]) +(ldx5[:,2] - ldx5[:,2].mean())**2 / np.sum((ldx5[:,2] - ldx5[:,2].mean())**2))



ax1_1,bx1_1=np.polyfit(xx[:], ldx1[:,0], deg=1)
ax1_2,bx1_2=np.polyfit(xx[:], ldx1[:,1], deg=1)
ax1_3,bx1_3=np.polyfit(xx[:], ldx1[:,2], deg=1)

ax2_1,bx2_1=np.polyfit(xx[:], ldx2[:,0], deg=1)
ax2_2,bx2_2=np.polyfit(xx[:], ldx2[:,1], deg=1)
ax2_3,bx2_3=np.polyfit(xx[:], ldx2[:,2], deg=1)

ax3_1,bx3_1=np.polyfit(xx[:], ldx3[:,0], deg=1)
ax3_2,bx3_2=np.polyfit(xx[:], ldx3[:,1], deg=1)
ax3_3,bx3_3=np.polyfit(xx[:], ldx3[:,2], deg=1)

ax4_1,bx4_1=np.polyfit(xx[:], ldx4[:,0], deg=1)
ax4_2,bx4_2=np.polyfit(xx[:], ldx4[:,1], deg=1)
ax4_3,bx4_3=np.polyfit(xx[:], ldx4[:,2], deg=1)

ax5_1,bx5_1=np.polyfit(xx[:], ldx5[:,0], deg=1)
ax5_2,bx5_2=np.polyfit(xx[:], ldx5[:,1], deg=1)
ax5_3,bx5_3=np.polyfit(xx[:], ldx5[:,2], deg=1)

yx1_est1 = ax1_1 * xx + bx1_1
yx1_est2= ax1_2 * xx + bx1_2
yx1_est3 = ax1_3 * xx + bx1_3

yx2_est1 = ax2_1 * xx + bx2_1
yx2_est2= ax2_2 * xx + bx2_2
yx2_est3 = ax2_3 * xx + bx2_3

yx3_est1 = ax3_1 * xx + bx3_1
yx3_est2= ax3_2 * xx + bx3_2
yx3_est3 = ax3_3 * xx + bx3_3

yx4_est1 = ax4_1 * xx + bx4_1
yx4_est2= ax4_2 * xx + bx4_2
yx4_est3 = ax4_3 * xx + bx4_3

yx5_est1 = ax5_1 * xx + bx5_1
yx5_est2= ax5_2 * xx + bx5_2
yx5_est3 = ax5_3 * xx + bx5_3

rx1_1=r2_score(ldx1[:,0],yx1_est1[:])
rx1_2=r2_score(ldx1[:,1],yx1_est2[:])
rx1_3=r2_score(ldx1[:,2],yx1_est3[:])

rx2_1=r2_score(ldx2[:,0],yx2_est1[:])
rx2_2=r2_score(ldx2[:,1],yx2_est2[:])
rx2_3=r2_score(ldx2[:,2],yx2_est3[:])

rx3_1=r2_score(ldx3[:,0],yx3_est1[:])
rx3_2=r2_score(ldx3[:,1],yx3_est2[:])
rx3_3=r2_score(ldx3[:,2],yx3_est3[:])

rx4_1=r2_score(ldx4[:,0],yx4_est1[:])
rx4_2=r2_score(ldx4[:,1],yx4_est2[:])
rx4_3=r2_score(ldx4[:,2],yx4_est3[:])

rx5_1=r2_score(ldx5[:,0],yx5_est1[:])
rx5_2=r2_score(ldx5[:,1],yx5_est2[:])
rx5_3=r2_score(ldx5[:,2],yx5_est3[:])


matplotlib.rcParams['xtick.direction'] = 'in'
y0=np.zeros((37))
y0[:]=0.0
dw=0.46
dh=0.16
ps1=[0.05,0.82,dw,dh]
ps3=[0.05,0.82-(0.02+dh),dw,dh]
ps5=[0.05,0.82-2*(0.02+dh),dw,dh]
ps7=[0.05,0.82-3*(0.02+dh),dw,dh]
ps9=[0.05,0.82-4*(0.02+dh),dw,dh]



ps2=[0.05+0.07+dw,0.82,dw,dh]
ps4=[0.05+0.07+dw,0.82-(0.02+dh),dw,dh]
ps6=[0.05+0.07+dw,0.82-2*(0.02+dh),dw,dh]
ps8=[0.05+0.07+dw,0.82-3*(0.02+dh),dw,dh]
ps10=[0.05+0.07+dw,0.82-4*(0.02+dh),dw,dh]


c1='blue'
c2='green'
c3='orange'
pt1=0.88
pt2=0.01
xminorLocator= MultipleLocator(1)
#fig=plt.figure(figsize=(16,10))
fig=plt.figure(figsize=(17,11))
host = host_subplot(521,position=ps1)
par = host.twinx()

# host.set_ylabel("Biomass anomaly",color='k',fontsize=18,fontname='Times New Roman')
# par.set_ylabel("NDVI anomaly",color='k',fontsize=18,fontname='Times New Roman')
p1, =host.plot(xx,ldx1[:,0],color=c1,marker='o')
p2, =host.plot(xx,ldx1[:,1],color=c2,marker='o')
p3, =par.plot(xx,ldx1[:,2],color=c3,marker='o')
host.plot(xx,yx1_est1,'--',color=c1)
host.plot(xx,yx1_est2,'--',color=c2)
par.plot(xx,yx1_est3,'--',color=c3)
x1 =np.linspace(1980, 2012, 17)
host.set_ylim(-3,3)
host.set_xlim(1980,2012)
host.set_xticks(x1)
host.tick_params(axis='y',colors='k')
par.set_ylim(-2,2)
par.tick_params(axis='y',colors='k')
host.set_xticklabels([ ])

labels1 = host.get_xticklabels() + host.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]

labels1 = par.get_xticklabels() + par.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]

host.annotate("(a)",xy=(pt2, pt1), xycoords='axes fraction',fontsize=16,fontname='Times New Roman')

host = host_subplot(522,position=ps2)

host.bar(xx,rdx1[:,2],width=0.4,label='ESM',color=c3)
host.bar(xx,rdx1[:,0],width=0.4,label='RF',color=c1)
host.bar(xx,rdx1[:,1],width=0.4,label='NDVI',color=c2)
x1 =np.linspace(1980, 2012, 17)

host.set_ylim(0,1.2)
host.set_xlim(1980,2012)
# host.set_yticks(y1)
host.set_xticks(x1)
host.set_xticklabels([ ])
labels1 = host.get_xticklabels() + host.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]
host.annotate("(f)",xy=(pt2, pt1), xycoords='axes fraction',fontsize=16,fontname='Times New Roman')

host = host_subplot(523,position=ps3)
par = host.twinx()
# host.set_ylabel("Biomass anomaly",color='k',fontsize=18,fontname='Times New Roman')
# par.set_ylabel("NDVI anomaly",color='k',fontsize=18,fontname='Times New Roman')
p1, =host.plot(xx,ldx2[:,0],color=c1,marker='o')
p2, =host.plot(xx,ldx2[:,1],color=c2,marker='o')
p3, =par.plot(xx,ldx2[:,2],color=c3,marker='o')
host.plot(xx,yx2_est1,'--',color=c1)
host.plot(xx,yx2_est2,'--',color=c2)
par.plot(xx,yx2_est3,'--',color=c3)
x1 =np.linspace(1980, 2012, 17)
host.set_ylim(-3,3)
host.set_xlim(1980,2012)
host.set_xticks(x1)
host.tick_params(axis='y',colors='k')
par.set_ylim(-2,2)
par.tick_params(axis='y',colors='k')
host.set_xticklabels([ ])
labels1 = host.get_xticklabels() + host.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]

labels1 = par.get_xticklabels() + par.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]
host.annotate("(b)",xy=(pt2, pt1), xycoords='axes fraction',fontsize=16,fontname='Times New Roman')

host = host_subplot(524,position=ps4)
host.bar(xx,rdx2[:,2],width=0.4,label='ESM',color=c3)
host.bar(xx,rdx2[:,0],width=0.4,label='RF',color=c1)
host.bar(xx,rdx2[:,1],width=0.4,label='NDVI',color=c2)
x1 =np.linspace(1980, 2012, 17)

host.set_ylim(0,0.6)
host.set_xlim(1980,2012)
# host.set_yticks(y1)
host.set_xticks(x1)
host.set_xticklabels([ ])
labels1 = host.get_xticklabels() + host.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]
host.annotate("(g)",xy=(pt2, pt1), xycoords='axes fraction',fontsize=16,fontname='Times New Roman')

host = host_subplot(525,position=ps5)
par = host.twinx()
host.set_ylabel("MTVCD anomaly",color='k',fontsize=18,fontname='Times New Roman')
par.set_ylabel("NDVI anomaly",color='k',fontsize=18,fontname='Times New Roman')
p1, =host.plot(xx,ldx3[:,0],color=c1,marker='o')
p2, =host.plot(xx,ldx3[:,1],color=c2,marker='o')
p3, =par.plot(xx,ldx3[:,2],color=c3,marker='o')
host.plot(xx,yx3_est1,'--',color=c1)
host.plot(xx,yx3_est2,'--',color=c2)
par.plot(xx,yx3_est3,'--',color=c3)
x1 =np.linspace(1980, 2012, 17)
host.set_ylim(-3,3)
host.set_xlim(1980,2012)
host.set_xticks(x1)
host.tick_params(axis='y',colors='k')
par.set_ylim(-2,2)
par.tick_params(axis='y',colors='k')
host.set_xticklabels([ ])
labels1 = host.get_xticklabels() + host.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]

labels1 = par.get_xticklabels() + par.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]
host.annotate("(c)",xy=(pt2, pt1), xycoords='axes fraction',fontsize=16,fontname='Times New Roman')

host = host_subplot(526,position=ps6)
host.bar(xx,rdx3[:,2],width=0.4,label='ESM',color=c3)
host.bar(xx,rdx3[:,0],width=0.4,label='RF',color=c1)
host.bar(xx,rdx3[:,1],width=0.4,label='NDVI',color=c2)
x1 =np.linspace(1980, 2012, 17)
host.set_ylabel("SD/Mean",color='k',fontsize=18,fontname='Times New Roman')
host.set_ylim(0,1.0)
host.set_xlim(1980,2012)
# host.set_yticks(y1)
host.set_xticks(x1)
host.set_xticklabels([ ])
labels1 = host.get_xticklabels() + host.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]
host.annotate("(h)",xy=(pt2, pt1), xycoords='axes fraction',fontsize=16,fontname='Times New Roman')

host = host_subplot(527,position=ps7)
par = host.twinx()
# host.set_ylabel("Biomass anomaly",color='k',fontsize=18,fontname='Times New Roman')
# par.set_ylabel("NDVI anomaly",color='k',fontsize=18,fontname='Times New Roman')
p1, =host.plot(xx,ldx4[:,0],color=c1,marker='o')
p2, =host.plot(xx,ldx4[:,1],color=c2,marker='o')
p3, =par.plot(xx,ldx4[:,2],color=c3,marker='o')
host.plot(xx,yx4_est1,'--',color=c1)
host.plot(xx,yx4_est2,'--',color=c2)
par.plot(xx,yx4_est3,'--',color=c3)
x1 =np.linspace(1980, 2012, 17)
host.set_ylim(-1.2,1.2)
host.set_xlim(1980,2012)
host.set_xticks(x1)
host.tick_params(axis='y',colors='k')
par.set_ylim(-1.2,1.2)
par.tick_params(axis='y',colors='k')
host.set_xticklabels([ ])
labels1 = host.get_xticklabels() + host.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]

labels1 = par.get_xticklabels() + par.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]
host.annotate("(d)",xy=(pt2, pt1), xycoords='axes fraction',fontsize=16,fontname='Times New Roman')

host = host_subplot(528,position=ps8)
host.bar(xx,rdx4[:,2],width=0.4,label='ESM',color=c3)
host.bar(xx,rdx4[:,0],width=0.4,label='RF',color=c1)
host.bar(xx,rdx4[:,1],width=0.4,label='NDVI',color=c2)
x1 =np.linspace(1980, 2012, 17)

host.set_ylim(0,2.0)
host.set_xlim(1980,2012)
# host.set_yticks(y1)
host.set_xticks(x1)
host.set_xticklabels([ ])
labels1 = host.get_xticklabels() + host.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]
host.annotate("(i)",xy=(pt2, pt1), xycoords='axes fraction',fontsize=16,fontname='Times New Roman')


host = host_subplot(529,position=ps9)
par = host.twinx()
# host.set_ylabel("Biomass anomaly",color='k',fontsize=18,fontname='Times New Roman')
# par.set_ylabel("NDVI anomaly",color='k',fontsize=18,fontname='Times New Roman')
p1, =host.plot(xx,ldx5[:,0],color=c1,marker='o')
p2, =host.plot(xx,ldx5[:,1],color=c2,marker='o')
p3, =par.plot(xx,ldx5[:,2],color=c3,marker='o')
host.plot(xx,yx5_est1,'--',color=c1)
host.plot(xx,yx5_est2,'--',color=c2)
par.plot(xx,yx5_est3,'--',color=c3)
x1 =np.linspace(1980, 2012, 17)
host.set_ylim(-2,2)
host.set_xlim(1980,2012)
host.set_xticks(x1)
host.tick_params(axis='y',colors='k')
par.set_ylim(-2,2)
par.tick_params(axis='y',colors='k')
host.tick_params(axis='x',pad=15.0)
labels1 = host.get_xticklabels() + host.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]

labels1 = par.get_xticklabels() + par.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]
host.annotate("(e)",xy=(pt2, pt1), xycoords='axes fraction',fontsize=16,fontname='Times New Roman')


host = host_subplot(520,position=ps10)
host.bar(xx,rdx5[:,2],width=0.4,label='ESM',color=c3)
host.bar(xx,rdx5[:,0],width=0.4,label='RF',color=c1)
host.bar(xx,rdx5[:,1],width=0.4,label='NDVI',color=c2)
x1 =np.linspace(1980, 2012, 17)

host.set_ylim(0,1.4)
host.set_xlim(1980,2012)
# host.set_yticks(y1)
host.set_xticks(x1)
labels1 = host.get_xticklabels() + host.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('12.0') for label in labels1]
host.annotate("(j)",xy=(pt2, pt1), xycoords='axes fraction',fontsize=16,fontname='Times New Roman')
host.tick_params(axis='x',pad=15.0)
plt.show()
fig.savefig(fdir+'Figure_biomass_class_trends_overlapped.jpg', dpi=300,bbox_inches = 'tight')
fig.clf()
print ("ok")
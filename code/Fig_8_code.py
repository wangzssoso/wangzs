# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:43:21 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:12:10 2021

@author: wangzs@igsnrr.ac.cn
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import  pandas  as pd
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['font.style']='normal'
matplotlib.rcParams['font.family']='Times New Roman'
def Fmm(t1):

    t2=('%.2f' % t1)
    return t2
def Ftm(t1):

    t2=('%.2f' % t1)
    return t2
def Gmm(t1):

    t2=('%.0f' % t1)
    return t2
def Gtm(t1):

    t2=('%.2f' % t1)
    return t2

# Fixing random state for reproducibility

fdir='I:/MsTMIP/MsTMP_China/bomes_plot_data/'
# Compute pie slices

finx=fdir+"year_r_RF_NDVI_ESM_data.dat"
dx1=np.loadtxt(finx,skiprows=1)
N = 29
sb1=dx1[:,1]
sb2=dx1[:,2]
ww1=dx1[:,1]
ww2=dx1[:,2]




FMEA1=Fmm(np.mean(sb1))
FMEA2=Fmm(np.mean(sb2))
FSDE1=Fmm(np.std(sb1))
FSDE2=Fmm(np.std(sb2))

FMES1=Ftm(np.mean(ww1))
FMES2=Ftm(np.mean(ww2))
FSDS1=Ftm(np.std(ww1))
FSDS2=Ftm(np.std(ww2))

# GMEA1=Gmm(np.mean(gw1))
# GMEA2=Gmm(np.mean(gw2))
# GSDE1=Gmm(np.std(gw1))
# GSDE2=Gmm(np.std(gw2))
# GMES1=Gtm(np.mean(tg1))
# GMES2=Gtm(np.mean(tg2))
# GSDS1=Gtm(np.std(tg1))
# GSDS2=Gtm(np.std(tg2))


width1=np.zeros((N))
width2=np.zeros((N))
twidth1=np.zeros((N))
twidth2=np.zeros((N))
rad1=np.zeros((N))
rad2=np.zeros((N))
grad1=np.zeros((N))
grad2=np.zeros((N))
mtg1=np.zeros((N))
mtg2=np.zeros((N))
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

for i in range(0,N):
    # mtg1[i]=tg1[i]*10.0
    # mtg2[i]=tg2[i]*10.0
    width1[i] = np.pi / 4 * (ww1[i]-0.7)
    width2[i] = np.pi / 4 * (ww2[i]-0.7)
    # twidth1[i] = np.pi / 4 * (mtg1[i]-0.1)
    # twidth2[i] = np.pi / 4 * (mtg2[i]-0.1)
    rad1[i] = 10 *(sb1[i]-0.75)
    rad2[i] = 10 *(sb2[i]-0.75)
    # grad1[i] = 10 *(gw1[i]/1000.0-0.3)
    # grad2[i] = 10 *(gw2[i]/1000.0-0.3)
print(width1.shape)
#tick_label=['     1982','     1990','1995','2000     ','2005     ','2010      ','2015','      2018']
tick_label=['1982','1983    ','1984     ','1985       ','1986      ','1987      ','1988      ','1989     ','1990      ','1991      ','1992      ','1993     ','1994     ','1995  ','1996','1997','   1998','   1999','      2000','      2001','      2002','     2003','    2004','     2005','     2006','     2007','     2008','     2009','     2010']
colors = plt.cm.viridis(rad1 / 10.)
fig= plt.figure(figsize=(16,12))
p1=[0.05,0.1,0.8,0.8]
# p2=[0.55,0.1,0.4,0.8]
# p3=[0.5,0.1,0.1,0.1]
# ax1=fig.add_axes(p1)
ax1 = plt.subplot(position=p1,projection='polar')

ax1.bar(theta, rad1, width=width2, bottom=0.2, color='green', alpha=0.99,label="MMRFE",tick_label=tick_label)
ax1.bar(theta, rad2, width=width1, bottom=0.2, color='orange', alpha=0.99,label="MMEM")
ax1.set_theta_direction(-1)
ax1.set_theta_zero_location("N",180.0)
#ax1.annotate('(a)',xy=(0.02,0.82),xycoords='figure fraction',fontsize=24,fontname='Times New Roman')
ax1.legend(frameon=False,bbox_to_anchor=[0.2,-0.150],borderaxespad=2,fontsize=20,ncol=2,loc=8)
labels1 = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
[label.set_fontsize('20.0') for label in labels1]
#ax1.set_title("FVC")
ax1.tick_params(axis='y',pad=30)

# ax2 = plt.subplot(position=p2,projection='polar')

# ax2.bar(theta, grad2, width=twidth2, bottom=0.2, color='green', alpha=0.99,tick_label=tick_label,label="With Agri. activity")
# ax2.bar(theta, grad1, width=twidth1, bottom=0.2, color='orange', alpha=0.99,label="Without Agri. activity")
# #ax2.annotate('(b)',xy=(0.52,0.82),xycoords='figure fraction',fontsize=24,fontname='Times New Roman')
# ax2.set_theta_direction(-1)
# ax2.set_theta_zero_location("N",180.0)
# labels1 = ax2.get_xticklabels() + ax2.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels1]
# [label.set_fontsize('20.0') for label in labels1]
#ax2.set_title("GPP")
# ax3=plt.subplot(position=p3)
# ax3.legend(frameon=False)
#ax1.annotate(Rname,xy=(0.45,0.85),xycoords='figure fraction',color='k',fontsize=24,fontname='Times New Roman')
# ax1.annotate('PCD',xy=(0.15,0.12),xycoords='figure fraction',color='k',fontsize=24,fontname='Times New Roman')
# ax1.annotate('Total Area (1×10$^{4}$ km$^{2}$)',xy=(0.28,0.12),xycoords='figure fraction',color='k',fontsize=24,fontname='Times New Roman')
# ax1.annotate('GPP (g C.m$^{-2}$) ',xy=(0.60,0.12),xycoords='figure fraction',color='k',fontsize=24,fontname='Times New Roman')
# ax1.annotate('Total GPP (Pg C.y$^{-1}$) ',xy=(0.78,0.12),xycoords='figure fraction',color='k',fontsize=24,fontname='Times New Roman')


ax1.annotate(str(FMEA1)+' $\pm$ '+str(FSDE1),xy=(0.50,0.03),xycoords='figure fraction',color='green',fontsize=24,fontname='Times New Roman')
# ax1.annotate(str(FMES2)+' $\pm$ '+str(FSDS2),xy=(0.32,0.08),xycoords='figure fraction',color='green',fontsize=24,fontname='Times New Roman')
# ax1.annotate(str(GMEA2)+' $\pm$ '+str(GSDE2),xy=(0.62,0.08),xycoords='figure fraction',color='green',fontsize=24,fontname='Times New Roman')
# ax1.annotate(str(GMES2)+' $\pm$ '+str(GSDS2),xy=(0.82,0.08),xycoords='figure fraction',color='green',fontsize=24,fontname='Times New Roman')

ax1.annotate(str(FMEA2)+' $\pm$ '+str(FSDE2),xy=(0.70,0.03),xycoords='figure fraction',color='orange',fontsize=24,fontname='Times New Roman')
# ax1.annotate(str(FMES1)+' $\pm$ '+str(FSDS1),xy=(0.32,0.04),xycoords='figure fraction',color='orange',fontsize=24,fontname='Times New Roman')
# ax1.annotate(str(GMEA1)+' $\pm$ '+str(GSDE1),xy=(0.62,0.04),xycoords='figure fraction',color='orange',fontsize=24,fontname='Times New Roman')
# ax1.annotate(str(GMES1)+' $\pm$ '+str(GSDS1),xy=(0.82,0.04),xycoords='figure fraction',color='orange',fontsize=24,fontname='Times New Roman')

#plt.subplots_adjust(top = 0.1, bottom = 0.1, hspace = 0, wspace = 0)
#plt.margins(0,0)
fig.savefig(fdir+'FVC_LUCC_NLUCC_29.jpg',dpi=300,bbox_inches='tight')
plt.show()
fig.clf()
print("ok")
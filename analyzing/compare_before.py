import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import LinearRegression
import scipy
import os
from os.path import join
import pandas as pd

datadir = './kcat/data/training_results_mh/'

import warnings
warnings.filterwarnings("ignore")

plt.style.use('./kcat/kcat_2024/analyzing/CCB_plot_style_0v4.mplstyle');
c_styles      = mpl.rcParams['axes.prop_cycle'].by_key()['color']   # fetch the defined color styles
high_contrast = ['#004488', '#DDAA33', '#BB5566', '#000000']

me_box_color = "blue"
old_box_color = "black"


str_fp_p_m,diff_fp_p_m,DRFP_p_m,ESM1b_p_m,ESM1b_ts_p_m,ESM1b_ts_DRFP_p_m,ESM1b_ts_DRFP_mean_p_m = [],[],[],[],[],[],[]
str_fp_p_h,diff_fp_p_h,DRFP_p_h,ESM1b_p_h,ESM1b_ts_p_h,ESM1b_ts_DRFP_p_h,ESM1b_ts_DRFP_mean_p_h = [],[],[],[],[],[],[]

str_fp_M_m,diff_fp_M_m,DRFP_M_m,ESM1b_M_m,ESM1b_ts_M_m,ESM1b_ts_DRFP_M_m,ESM1b_ts_DRFP_mean_M_m = [],[],[],[],[],[],[]
str_fp_M_h,diff_fp_M_h,DRFP_M_h,ESM1b_M_h,ESM1b_ts_M_h,ESM1b_ts_DRFP_M_h,ESM1b_ts_DRFP_mean_M_h = [],[],[],[],[],[],[]

str_fp_R_m,diff_fp_R_m,DRFP_R_m,ESM1b_R_m,ESM1b_ts_R_m,ESM1b_ts_DRFP_R_m,ESM1b_ts_DRFP_mean_R_m = [],[],[],[],[],[],[]
str_fp_R_h,diff_fp_R_h,DRFP_R_h,ESM1b_R_h,ESM1b_ts_R_h,ESM1b_ts_DRFP_R_h,ESM1b_ts_DRFP_mean_R_h = [],[],[],[],[],[],[]

result_model = ['str_fp','diff_fp','DRFP','ESM1b','ESM1b_ts','ESM1b_ts_DRFP','ESM1b_ts_DRFP_mean']
labs = ['str_fp_me','str_fp_author','diff_fp_me','diff_fp_author','DRFP_me','DRFP_author','ESM1b_me','ESM1b_author','ESM1b_ts_me','ESM1b_ts_author','ESM1b_ts_DRFP_me','ESM1b_ts_DRFP_author','ESM1b_ts_DRFP_mean_me','ESM1b_ts_DRFP_mean_author']

#绘制person图

#读取结果
result_y = 'p'


for model in result_model:
    listname = model+'_'+result_y+'_m'
    with open(datadir+model+result_y+"m.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            globals()[listname].append(float(line.strip()))
    
    listname = model+'_'+result_y+'_h'
    with open(datadir+model+result_y+"h.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            globals()[listname].append(float(line.strip()))


fig, ax = plt.subplots(figsize= (20,8))


plt.rcParams.update({'font.size': 28})
plt.ylim(0.1, 0.8)
plt.xlim(0.5, 15 + 0.5)



Boxplots = []
ticks = []
i=0
for model in result_model:
    listname_m = model+'_'+result_y+'_m'
    result_m = globals()[listname_m]
    listname_h = model+'_'+result_y+'_h'
    result_h = globals()[listname_h]
        
    Boxplots.append(result_m)
    Boxplots.append(result_h)
    ticks.append(i+1)
    ticks.append(i+2)
    i = i+2



# 绘制箱图并设置箱体颜色
for i, box_data in enumerate(Boxplots):
    box_color = me_box_color if i % 2 == 0 else old_box_color
    plt.boxplot(box_data, positions=[ticks[i]], widths=0.6,
                medianprops={"linewidth": 2, "solid_capstyle": "butt", "c": "darkred"},
                boxprops={"linewidth": 1.5, "solid_capstyle": "butt", "color": box_color},
                whiskerprops={"linewidth": 1.5, "solid_capstyle": "butt"},
                capprops={"linewidth": 1.5, "solid_capstyle": "butt"})




ax.locator_params(axis="y", nbins=8)

ticks1 = ticks
ax.set_xticks(ticks1)
ax.set_xticklabels([])
ax.tick_params(axis='x', which="major", length=10)
ax.tick_params(axis='y', length=10)
#ax.locator_params(axis="y", nbins=4)


ticks2 = list(np.array(ticks)-0.01)

ax.set_xticks(ticks2, minor=True)
ax.set_xticklabels(labs, minor=True, y= -0.03, fontsize = 22)
ax.tick_params(axis='x', which="minor",length=0, rotation = 60)
#loc = plticker.MultipleLocator(base=0.02) # this locator puts ticks at regular intervals
#ax.yaxis.set_major_locator(loc)

plt.ylabel("Pearson r")
ax.yaxis.set_label_coords(-0.18, 0.5)
#plt.legend(loc = "upper right")

plt.savefig(datadir+"Person.png")







#绘制MSE图


result_y = 'M'


for model in result_model:
    listname = model+'_'+result_y+'_m'
    with open(datadir+model+result_y+"m.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            globals()[listname].append(float(line.strip()))
    
    listname = model+'_'+result_y+'_h'
    with open(datadir+model+result_y+"h.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            globals()[listname].append(float(line.strip()))


fig, ax = plt.subplots(figsize= (20,8))


plt.rcParams.update({'font.size': 28})
plt.ylim(0.5, 1.4)
plt.xlim(0.5, 15 + 0.5)



Boxplots = []
ticks = []
i=0
for model in result_model:
    listname_m = model+'_'+result_y+'_m'
    result_m = globals()[listname_m]
    listname_h = model+'_'+result_y+'_h'
    result_h = globals()[listname_h]
        
    Boxplots.append(result_m)
    Boxplots.append(result_h)
    ticks.append(i+1)
    ticks.append(i+2)
    i = i+2

        
for i, box_data in enumerate(Boxplots):
    box_color = me_box_color if i % 2 == 0 else old_box_color
    plt.boxplot(box_data, positions=[ticks[i]], widths=0.6,
                medianprops={"linewidth": 2, "solid_capstyle": "butt", "c": "darkred"},
                boxprops={"linewidth": 1.5, "solid_capstyle": "butt", "color": box_color},
                whiskerprops={"linewidth": 1.5, "solid_capstyle": "butt"},
                capprops={"linewidth": 1.5, "solid_capstyle": "butt"})




ax.locator_params(axis="y", nbins=8)

ticks1 = ticks
ax.set_xticks(ticks1)
ax.set_xticklabels([])
ax.tick_params(axis='x', which="major", length=10)
ax.tick_params(axis='y', length=10)
#ax.locator_params(axis="y", nbins=4)


ticks2 = list(np.array(ticks)-0.01)

ax.set_xticks(ticks2, minor=True)
ax.set_xticklabels(labs, minor=True, y= -0.03, fontsize = 22)
ax.tick_params(axis='x', which="minor",length=0, rotation = 60)
#loc = plticker.MultipleLocator(base=0.02) # this locator puts ticks at regular intervals
#ax.yaxis.set_major_locator(loc)

plt.ylabel("MSE")
ax.yaxis.set_label_coords(-0.13, 0.5)
#plt.legend(loc = "upper right")

plt.savefig(datadir+"MSE.png")











#绘制R2图

result_y = 'R'


for model in result_model:
    listname = model+'_'+result_y+'_m'
    with open(datadir+model+result_y+"m.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            globals()[listname].append(float(line.strip()))
    
    listname = model+'_'+result_y+'_h'
    with open(datadir+model+result_y+"h.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            globals()[listname].append(float(line.strip()))


fig, ax = plt.subplots(figsize= (20,8))


plt.rcParams.update({'font.size': 28})
plt.ylim(-0.01, 0.5)
plt.xlim(0.5, 15 + 0.5)



Boxplots = []
ticks = []
i=0
for model in result_model:
    listname_m = model+'_'+result_y+'_m'
    result_m = globals()[listname_m]
    listname_h = model+'_'+result_y+'_h'
    result_h = globals()[listname_h]
        
    Boxplots.append(result_m)
    Boxplots.append(result_h)
    ticks.append(i+1)
    ticks.append(i+2)
    i = i+2

        
for i, box_data in enumerate(Boxplots):
    box_color = me_box_color if i % 2 == 0 else old_box_color
    plt.boxplot(box_data, positions=[ticks[i]], widths=0.6,
                medianprops={"linewidth": 2, "solid_capstyle": "butt", "c": "darkred"},
                boxprops={"linewidth": 1.5, "solid_capstyle": "butt", "color": box_color},
                whiskerprops={"linewidth": 1.5, "solid_capstyle": "butt"},
                capprops={"linewidth": 1.5, "solid_capstyle": "butt"})




ax.locator_params(axis="y", nbins=8)

ticks1 = ticks
ax.set_xticks(ticks1)
ax.set_xticklabels([])
ax.tick_params(axis='x', which="major", length=10)
ax.tick_params(axis='y', length=10)
#ax.locator_params(axis="y", nbins=4)


ticks2 = list(np.array(ticks)-0.01)

ax.set_xticks(ticks2, minor=True)
ax.set_xticklabels(labs, minor=True, y= -0.03, fontsize = 22)
ax.tick_params(axis='x', which="minor",length=0, rotation = 60)
#loc = plticker.MultipleLocator(base=0.02) # this locator puts ticks at regular intervals
#ax.yaxis.set_major_locator(loc)

plt.ylabel("R2")
ax.yaxis.set_label_coords(-0.13, 0.5)
#plt.legend(loc = "upper right")

plt.savefig(datadir+"R2.png")










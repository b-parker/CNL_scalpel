import os,sys
import re
from numpy import *
import pandas as pd
from scipy import stats
from numpy import *

def getData(subs,hemi):
    data_all = []
    for sub in subs:
        ses = sub[-1]
        dfile = '/home/weiner/shakki/NotBackedUp/spatial_autocorrelation/%s/labelnw/adj-labels-%s.txt'%(sub,hemi)
        print(dfile)
        if os.path.exists(dfile):
            df_sub = pd.read_csv(dfile,sep='\s+',header=None)
            df_sub = df_sub.round(decimals=3)
            if not any(isnan(df_sub)):
                df_sub[isinf(df_sub)] = 0
                data_all.append(df_sub)

    return data_all


def saveCohortData(data_lh,data_rh,savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # save mean data
    n_subs = shape(data_lh)[0]
    n_roi = shape(data_lh)[1]
    all_vals = zeros((2*n_subs,n_roi,n_roi))
    for i in arange(n_subs):
        all_vals[i,:,:] = array(data_lh[i])
        all_vals[i+n_subs,:] = array(data_rh[i])
    mean_data = squeeze(nanmean(all_vals,0))
    savetxt('%s/mean_adj.csv'%(savedir),mean_data,delimiter=',',header='',fmt='%.04f')

    min_data = squeeze(amin(all_vals,0))
    savetxt('%s/min_adj.csv'%(savedir),mean_data,delimiter=',',header='',fmt='%.04f')

    from copy import deepcopy
    mean_data_bin = deepcopy(mean_data)
    mean_data_bin[mean_data > 5] = 0
    mean_data_bin[mean_data < 5] = 1
    savetxt('%s/min_adj_bin.csv'%(savedir),min_data_bin,delimiter=',',header='',fmt='%.04f')

    max_data = squeeze(amax(all_vals,0))
    savetxt('%s/max_adj.csv'%(savedir),mean_data,delimiter=',',header='',fmt='%.04f')


if __name__ == '__main__':
    # list of potential subjects
    subs = loadtxt('/home/weiner/shakki/scripts/fc_nw_motioncorrect/fc/namelist.csv',dtype='str')

    # filter out subjects without connectivity values on requires labels
    data_lh = getData(subs,'lh')
    data_rh = getData(subs,'rh')

    # save
    saveCohortData(data_lh,data_rh,os.getcwd())


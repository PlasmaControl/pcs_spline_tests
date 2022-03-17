import matplotlib.pyplot as plt
import numpy as np
import toksearch

from pcs_fit_helpers import spline_eval

# BASIC TEST:
psinspline=[0,0.01,0.6,0.8]
mhatspline=[0,0.1,1,2]
num_cer_points=3
NFIT=121

# BIGGER TEST:
shot=936935 #936935, 187076
time=629

ftssmhat_data=[]
times=[]
ftsspsin_data=[]
ftssn_data=toksearch.PtDataSignal('ftssn').fetch(shot)['data']
for i in range(1,76):
    sig=toksearch.PtDataSignal(f'ftssmhat{i}').fetch(shot)
    ftssmhat_data.append(sig['data'])
    sig=toksearch.PtDataSignal(f'ftsspsin{i}').fetch(shot)
    ftsspsin_data.append(sig['data'])
times=sig['times']
ftssmhat_data=np.array(ftssmhat_data)
ftsspsin_data=np.array(ftsspsin_data)

# 0 index here is just because nonzero returns a tuple of arrays (squeeze)
nonzero_time_inds=np.nonzero(ftssn_data)[0]
nonzero_times=times[nonzero_time_inds]
nearest_available_time=nonzero_times[np.argmin(np.abs(nonzero_times-time))]
time_ind=np.argmin(np.abs(times-nearest_available_time))
# ftssn is the number of cer fit points at the timeslice
num_cer_points=int(ftssn_data[time_ind])
psinspline=ftsspsin_data[:num_cer_points,time_ind]
mhatspline=ftssmhat_data[:num_cer_points,time_ind]

# ftx signals only available in recent PCS versions
if shot>190000:
    ftxpr=toksearch.PtDataSignal('ftxpr').fetch(shot)
    ftxpr['data']=ftxpr['data'].reshape((-1,121))
    ftxpr['times']=ftxpr['times'][:ftxpr['data'].shape[0]]

    ftx_time_ind=np.argmin(np.abs(ftxpr['times']-nearest_available_time))
    plt.plot(np.linspace(0,1.2,NFIT),ftxpr['data'][ftx_time_ind,:],label='online version')

v=spline_eval(psinspline,mhatspline,num_cer_points,NFIT)
plt.plot(np.linspace(0,1.2,NFIT),v,label='offline version',linestyle='--')

plt.scatter(psinspline,
            mhatspline,
            marker='x',label='data to fit',c='k')

plt.title(f'Rotation fits at {nearest_available_time:.0f} ms, shot {shot}')
plt.xlabel('psin')
plt.legend()
plt.show()

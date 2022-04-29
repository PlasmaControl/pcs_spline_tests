import matplotlib.pyplot as plt
import numpy as np
import toksearch

from pcs_fit_helpers import spline_eval, calculate_mhat

# BASIC TEST:
psinspline=[0,0.01,0.6,0.8]
mhatspline=[0,0.1,1,2]
num_cer_points=3
NFIT=121

# BIGGER TEST:
shot=936935 #936935, 187076
time=629

ftssmhat_data=[]
ftssrot_data=[]
times=[]
ftsspsin_data=[]
ftssn_data=toksearch.PtDataSignal('ftssn').fetch(shot)['data']
for i in range(1,76):
    sig=toksearch.PtDataSignal(f'ftssmhat{i}').fetch(shot)
    ftssmhat_data.append(sig['data'])
    sig=toksearch.PtDataSignal(f'ftsspsin{i}').fetch(shot)
    ftsspsin_data.append(sig['data'])
    sig=toksearch.PtDataSignal(f'ftssrot{i}').fetch(shot)
    ftssrot_data.append(sig['data'])

times=sig['times']
ftssmhat_data=np.array(ftssmhat_data)
ftsspsin_data=np.array(ftsspsin_data)
ftssrot_data=np.array(ftssrot_data)

# 0 index here is just because nonzero returns a tuple of arrays (squeeze)
nonzero_time_inds=np.nonzero(ftssn_data)[0]
nonzero_times=times[nonzero_time_inds]
nearest_available_time=nonzero_times[np.argmin(np.abs(nonzero_times-time))]
time_ind=np.argmin(np.abs(times-nearest_available_time))
# ftssn is the number of cer fit points at the timeslice
num_cer_points=int(ftssn_data[time_ind])
psinspline=ftsspsin_data[:num_cer_points,time_ind]
mhatspline=ftssmhat_data[:num_cer_points,time_ind]
srot=ftssrot_data[:num_cer_points,time_ind]

# ftx signals only available in recent PCS versions
if False: #shot>190000:
    ftxpr=toksearch.PtDataSignal('ftxpr').fetch(shot)
    ftxpr['data']=ftxpr['data'].reshape((-1,121))
    ftxpr['times']=ftxpr['times'][:ftxpr['data'].shape[0]]

    ftx_time_ind=np.argmin(np.abs(ftxpr['times']-nearest_available_time))
    plt.plot(np.linspace(0,1.2,NFIT),ftxpr['data'][ftx_time_ind,:],label='online fit',c='k')

v=spline_eval(psinspline,mhatspline,num_cer_points,NFIT)
plt.scatter(psinspline,
            srot,
            label='online data',c='k')
plt.scatter(psinspline,
            mhatspline,
            marker='x',label='online mhat',c='k')
plt.plot(np.linspace(0,1.2,NFIT),v,label='online fit',c='k')

try:
    import h5py
    with h5py.File('/home/abbatej/tokamak-transport/data.h5') as f:
        times=f['times']
        time_ind=np.searchsorted(times,time)
        psi=f['187076']['cer_rot_psi_raw_1d'][time_ind]
        rot=f['187076']['cer_rot_raw_1d'][time_ind]*1e3
        (mPsi,mHat)=calculate_mhat(psi,rot,p=0.5,dxMin=0.01)
        mPsi=mPsi[:-1] # still not sure why the last index is always 0, should talk to ricardo (TODO)
        mHat=mHat[:-1]
        splined_rot=spline_eval(mPsi,mHat,len(mHat))
        plt.scatter(psi, rot, label='offline data',c='r')
        plt.scatter(mPsi, mHat, label='offline mhat', marker='x',c='r')
        plt.plot(np.linspace(0,1.2,NFIT), splined_rot, label='offline fit',c='r')
except:
    print("Use tmp.yaml for new_database_maker.py from PlasmaControl/data-fetching repo to read in raw CER rotation from the base shot associated with your sim")
plt.title(f'Rotation fits at {nearest_available_time:.0f} ms, shot {shot}')
plt.xlabel('psin')
plt.legend()
plt.show()

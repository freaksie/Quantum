import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture

# common config
LO_start=1000  #Neel : Start point of datastream
LO_stop=2500   #Neel : End point ie 2.74µs
time=np.arange(0e-9,4096e-9,1e-9)
# IF_freq_dict={'Q1':-92603821.33982977,'Q2':-35632930.91206281,'Q5':130603855.96404536,'Q6':190987366.095816}
IF_freq_dict={'Q3' :17597366.299644876 }

for qid in ['Q3']:
	# import data
	Id_file=glob.glob('/home/nxv8988/Neel/Quantum/20210325/Q3/meas/_readouttrajectory_I__opsel2_mon_sel00_mon_sel11_avrfactor1_qubitidQ3_qubitidreadQ3_*.dat')
	X_file=glob.glob('/home/nxv8988/Neel/Quantum/20210325/Q3/meas/_readouttrajectory_X180__opsel2_mon_sel00_mon_sel11_avrfactor1_qubitidQ3_qubitidreadQ3_*.dat')
	Id_I_list=np.array([])
	Id_Q_list=np.array([])
	
	for f in Id_file:
		with open(f,'r') as dat:
			temp=json.load(dat)
		Id_I_list=np.append(Id_I_list,temp['buf0'])
		Id_Q_list=np.append(Id_Q_list,temp['buf1'])
	print(Id_I_list.shape)
	Id_I_list=Id_I_list.reshape(len(Id_file),len(time))
	Id_Q_list=Id_Q_list.reshape(len(Id_file),len(time))
	X_I_list=np.array([])
	X_Q_list=np.array([])
	for f in X_file:
		with open(f,'r') as dat:
			temp=json.load(dat)
		X_I_list=np.append(X_I_list,temp['buf0'])
		X_Q_list=np.append(X_Q_list,temp['buf1'])
	print(X_I_list.shape)
	X_I_list=X_I_list.reshape(len(Id_file),len(time))
	X_Q_list=X_Q_list.reshape(len(Id_file),len(time))

	## rotate IQ according to readout frequency
	IF_freq=IF_freq_dict[qid]
	IF_rot=np.exp(-2j*np.pi*IF_freq*time)#N eel : New wave with frequency IF_Freq that include complex component
	np.save('ifrot.npy',IF_rot)
	# Neel : (Id_I_list[k]+1j*Id_Q_list[k]) : for state 0, combine I as real and Q as imaginary part of signal
	Id_list=np.array([(Id_I_list[k]+1j*Id_Q_list[k])*IF_rot for k in range(len(Id_I_list))])
	X_list=np.array([(X_I_list[k]+1j*X_Q_list[k])*IF_rot for k in range(len(Id_I_list))])

	## data averaging Neel: Averaging through all shots.
	Id_avg=np.mean(Id_list,axis=0)
	X_avg=np.mean(X_list,axis=0)
	# exponentially weighted moving average
	avg_elements=200
	Id_I_smooth=pd.DataFrame({'A': np.real(Id_avg)}).ewm(span=avg_elements).mean().to_numpy()
	Id_Q_smooth=pd.DataFrame({'A': np.imag(Id_avg)}).ewm(span=avg_elements).mean().to_numpy()
	X_I_smooth=pd.DataFrame({'A': np.real(X_avg)}).ewm(span=avg_elements).mean().to_numpy()
	X_Q_smooth=pd.DataFrame({'A': np.imag(X_avg)}).ewm(span=avg_elements).mean().to_numpy()

	# time plot |0> state
	pt,ax=plt.subplots(1,2,figsize=(8,3))
	ax[0].plot(time*1e6,np.real(Id_avg),color='b')
	ax[0].plot(time*1e6,Id_I_smooth,color='w')
	ax[0].set_xlabel('time (us)')
	ax[0].set_ylabel('<I> (arb.)')
	ax[0].set_title(qid+' raw and average: I data |0> state')
	ax[1].plot(time*1e6,np.imag(Id_avg),color='b')
	ax[1].plot(time*1e6,Id_Q_smooth,color='w')
	ax[1].set_xlabel('time (us)')
	ax[1].set_ylabel('<Q> (arb.)')
	ax[1].set_title(qid+' raw and average: Q data |0> state')
	plt.tight_layout()
	plt.savefig(qid+'_raw_state0.png')

	# time plot |1> state
	pt,ax=plt.subplots(1,2,figsize=(8,3))
	ax[0].plot(time*1e6,np.real(X_avg),color='r')
	ax[0].plot(time*1e6,X_I_smooth,color='w')
	ax[0].set_xlabel('time (us)')
	ax[0].set_ylabel('<I> (arb.)')
	ax[0].set_title(qid+' raw and average: I data |1> state')
	ax[1].plot(time*1e6,np.imag(X_avg),color='r')
	ax[1].plot(time*1e6,X_Q_smooth,color='w')
	ax[1].set_xlabel('time (us)')
	ax[1].set_ylabel('<Q> (arb.)')
	ax[1].set_title(qid+' raw and average: Q data |1> state')
	plt.tight_layout()
	plt.savefig(qid+'_raw_state1.png')

	# trajectory
	plt.figure()
	traj_step=50
	plt.plot(Id_I_smooth[LO_start:LO_stop:traj_step],Id_Q_smooth[LO_start:LO_stop:traj_step], linestyle='-', marker='.',color='b',label='|0>')
	#plt.plot(Id_I_smooth[LO_start:(LO_start+500):20],Id_Q_smooth[LO_start:(LO_start+500):20], linestyle='-', marker='.',color='b',label='|0>')
	plt.plot(Id_I_smooth[LO_start],Id_Q_smooth[LO_start],marker='X',color='b')
	plt.plot(X_I_smooth[LO_start:LO_stop:traj_step],X_Q_smooth[LO_start:LO_stop:traj_step], linestyle='-', marker='.',color='r',label='|1>')
	#plt.plot(X_I_smooth[LO_start:(LO_start+500):20],X_Q_smooth[LO_start:(LO_start+500):20], linestyle='-', marker='.',color='r',label='|1>')
	plt.plot(X_I_smooth[LO_start],X_Q_smooth[LO_start],marker='X',color='r')
	plt.xlabel('<I> (arb.)')
	plt.ylabel('<Q> (arb.)')
	plt.legend()
	plt.title(qid+' trajectory')
	plt.savefig(qid+'_trajectory.png')
	print('|0>',Id_I_smooth[LO_start:LO_stop:50],Id_Q_smooth[LO_start:LO_stop:50])
	print('|1>',X_I_smooth[LO_start:LO_stop:50],X_Q_smooth[LO_start:LO_stop:50])

	# weighting function
	weight_I=np.transpose(abs(Id_I_smooth-X_I_smooth))[0]
	weight_Q=np.transpose(abs(Id_Q_smooth-X_Q_smooth))[0]
	plt.figure()
	plt.plot(time*1e6,weight_I,label=r'|$\ \!$I$_{|1\rangle}$-I$_{|0\rangle}\ \!$|')
	plt.plot(time*1e6,weight_Q,label=r'|$\ \!$Q$_{|1\rangle}$-Q$_{|0\rangle}\ \!$|')
	plt.xlabel('time (us)')
	plt.ylabel('amplitude (arb.)')
	plt.legend()
	#plt.ylim([0,1000])
	plt.title(qid+' weighting function')
	plt.savefig(qid+'_weighting_function.png')

	# weighted scatter
	weighted_Id_I=np.mean(np.real(Id_list[:,LO_start:LO_stop])*weight_I[LO_start:LO_stop],axis=1)*10000
	weighted_Id_Q=np.mean(np.imag(Id_list[:,LO_start:LO_stop])*weight_Q[LO_start:LO_stop],axis=1)*10000
	weighted_X_I=np.mean(np.real(X_list[:,LO_start:LO_stop])*weight_I[LO_start:LO_stop],axis=1)*10000
	weighted_X_Q=np.mean(np.imag(X_list[:,LO_start:LO_stop])*weight_Q[LO_start:LO_stop],axis=1)*10000
	plt.figure()
	plt.scatter(weighted_Id_I,weighted_Id_Q,color='b')
	plt.scatter(weighted_X_I,weighted_X_Q,color='r')
	plt.xlabel('<I> (arb.)')
	plt.ylabel('<Q> (arb.)')
	plt.title(qid+' weighted scatter')
	plt.savefig(qid+'_weighted_scatter.png')
	X=list(zip(np.append(weighted_Id_I,weighted_X_I),np.append(weighted_Id_Q,weighted_X_Q)))
	gmm_X=GaussianMixture(n_components=2,covariance_type='full').fit(X)
	bitstring=np.split(gmm_X.predict(X),2)
	print('P(0|0)=',len(bitstring[0][bitstring[0]==0])/len(bitstring[0]))
	print('P(1|1)=',len(bitstring[1][bitstring[1]==1])/len(bitstring[1]))
	X=list(zip(weighted_Id_I,weighted_Id_Q))
	gmm_X_1=GaussianMixture(n_components=1,covariance_type='spherical').fit(X)
	X=list(zip(weighted_X_I,weighted_X_Q))
	gmm_X_2=GaussianMixture(n_components=1,covariance_type='spherical').fit(X)

	# unweighted scatter
	unweighted_Id_I=np.mean(np.real(Id_list[:,LO_start:LO_stop]),axis=1)
	unweighted_Id_Q=np.mean(np.imag(Id_list[:,LO_start:LO_stop]),axis=1)
	unweighted_X_I=np.mean(np.real(X_list[:,LO_start:LO_stop]),axis=1)
	unweighted_X_Q=np.mean(np.imag(X_list[:,LO_start:LO_stop]),axis=1)	
	print("-----------Shape----------",unweighted_Id_I.shape)
	plt.figure()
	plt.scatter(unweighted_Id_I,unweighted_Id_Q,color='b')
	plt.scatter(unweighted_X_I,unweighted_X_Q,color='r')
	plt.xlabel('<I> (arb.)')
	plt.ylabel('<Q> (arb.)')
	plt.title(qid+' unweighted scatter')
	plt.savefig(qid+'_unweighted_scatter.png')
	Y=list(zip(np.append(unweighted_Id_I,unweighted_X_I),np.append(unweighted_Id_Q,unweighted_X_Q)))
	gmm_Y=GaussianMixture(n_components=2,covariance_type='full').fit(Y)
	bitstring=np.split(gmm_Y.predict(Y),2)
	print('P(0|0)=',len(bitstring[0][bitstring[0]==0])/len(bitstring[0]))
	print('P(1|1)=',len(bitstring[1][bitstring[1]==1])/len(bitstring[1]))
	X=list(zip(unweighted_Id_I,unweighted_Id_Q))
	gmm_Y_1=GaussianMixture(n_components=1,covariance_type='spherical').fit(X)
	X=list(zip(unweighted_X_I,unweighted_X_Q))
	gmm_Y_2=GaussianMixture(n_components=1,covariance_type='spherical').fit(X)

	print((np.sqrt(np.sum((gmm_X_1.means_[0]-gmm_X_2.means_[0])**2)))/(np.sqrt(gmm_X_1.covariances_[0])+np.sqrt(gmm_X_2.covariances_[0])))
	print((np.sqrt(np.sum((gmm_Y_1.means_[0]-gmm_Y_2.means_[0])**2)))/(np.sqrt(gmm_Y_1.covariances_[0])+np.sqrt(gmm_Y_2.covariances_[0])))

plt.show()


## train 

import numpy as np

Ytrain = np.genfromtxt('Ytrain.txt')
Ytest = np.genfromtxt('Ytest.txt')
wtrain = np.genfromtxt('wtrain.txt')
wtest = np.genfromtxt('wtest.txt')

## normalize:

Ytrain = Ytrain.astype('float')
Ytest = Ytest.astype('float')
Ytrain /= Ytrain.max()
Ytest /= Ytest.max()

###############################################################
## define gradient descent and error functions

## weird error: omp_get_num_procs
## python3 doesn't have this problem ...

def du(lmbda, U, V, Y, w, i):

	Sum = 0.
	for j in range(len(V)):
		Sum += V[j] *  (Y[i,j] - float(U[i] * V[j].T)) * w[i,j]

	return lmbda * U[i] - Sum

def dv(lmbda, U, V, Y, w, j):

	Sum = 0.
	for i in range(len(U)):
		Sum += U[i] * (Y[i,j] - float(U[i] * V[j].T)) * w[i,j]	

	return lmbda * V[j] - Sum

def E(U,V,Y,w):
	M = U.shape[0]
	K = U.shape[1]
	return np.sum( w*np.array((Y - U*V.T))**2. ) / np.sum(w)

###############################################################

K = 20	## given in assignment, don't vary
M,N = Ytrain.shape
lmbda = 30.	## adjust manually
eta = 1.E-3	## adjust manually

U = np.matrix( np.random.rand(M,K) ) - 0.5
V = np.matrix( np.random.rand(N,K) ) - 0.5

maxIters = 1.E5
Eins = np.zeros(maxIters)
Eouts = np.zeros(maxIters)
Eins[0] = E(U,V,Ytrain,wtrain)
Eouts[0] = E(U,V,Ytest,wtest)

stopCond = False
nIter = 0

while stopCond == False:

	nIter += 1
	
	if nIter == maxIters - 1: stopCond = True

	i = np.random.randint(len(U))
	U[i] = U[i] - eta * du(lmbda,U,V,Ytrain,wtrain,i)

	j = np.random.randint(len(V))	
	V[j] = V[j] - eta * dv(lmbda,U,V,Ytrain,wtrain,j)

	Eins[nIter] = E(U,V,Ytrain,wtrain)
	Eouts[nIter] = E(U,V,Ytest,wtest)

	if nIter % 10 == 0:
	
		if Eins[nIter-100] < Eins[nIter] and nIter >= 1000: stopCond = True

		## print updates to errors
		print('%.1E\t%d\tE_in: %.1f\tE_out: %.1f\n' % (lmbda,nIter,Eins[nIter],Eouts[nIter]) )
	
Eins = Eins[:nIter+1]
Eouts = Eouts[:nIter+1]

doc = {		
	   'Ein': Eins,
	   'Eout': Eouts,
	   'lambda': lmbda,
	   'K': K,
	   'eta': eta,
	   'U': U,
	   'V': V
	   }

import pickle
docs = pickle.load(open('train_docs.pkl','rb'))
docs.append(doc)
pickle.dump(docs,open('train_docs.pkl','wb'))

import matplotlib.pyplot as plt
fig=plt.figure()
plt.plot(Eins / np.sum(wtrain))
plt.plot(Eouts / np.sum(wtest))
plt.plot((Eins / np.sum(wtrain))/(Eouts / np.sum(wtest)) )
fig.show()



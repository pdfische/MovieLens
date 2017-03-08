
## machine learning, project 3
## preprocess/format data to train SVD

import numpy as np

###############################################################
## read data

data = np.genfromtxt('./data.txt')	## columns = user id, movie id, rating

users = list(set(data[:,0]))
movies = list(set(data[:,1]))

###############################################################
## format data

M = len(users)
N = len(movies)

Ytrain = np.zeros((M,N))
Ytest = np.zeros((M,N))
wtrain = np.zeros((M,N))
wtest = np.zeros((M,N))

## make dictionary of users / movies / ratings:

docs = {}

for i in range(len(data)):
	user,movie,rating = data[i].astype('int')
	if user in docs.keys():
		docs[user][movie] = rating	
	else:
		docs[user] = {movie: rating}

## fill in ratings to Y and 1/0 to weights:

for m in range(M):
	for n in range(N):
		
		if movies[n] in docs[ users[m] ].keys():		## user gave movie a rating

			## ~80% goes to training set
			if np.random.rand() < 0.8:
				Ytrain[m][n] = docs[ users[m] ][ movies[n] ]
				wtrain[m][n] = 1
			## ~20% goes to test set
			else:
				Ytest[m][n] = docs[ users[m] ][ movies[n] ]
				wtest[m][n] = 1
			
		else: pass		## no rating, leave weight at 0
		

###############################################################

np.savetxt('Ytrain.txt',Ytrain,delimiter='\t',fmt='%d')
np.savetxt('Ytest.txt',Ytest,delimiter='\t',fmt='%d')
np.savetxt('wtrain.txt',wtrain,delimiter='\t',fmt='%d')
np.savetxt('wtest.txt',wtest,delimiter='\t',fmt='%d')




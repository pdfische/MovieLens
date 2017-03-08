
import numpy as np
import matplotlib.pyplot as plt

######################################################
## load data:

Ytrain = np.genfromtxt('Ytrain.txt')
Ytest = np.genfromtxt('Ytest.txt')
wtrain = np.genfromtxt('wtrain.txt')
wtest = np.genfromtxt('wtest.txt')

Y = Ytrain + Ytest
w = wtrain + wtest

movies = np.genfromtxt('movies.txt',delimiter='\t',dtype='str')

######################################################
## distribution of all ratings:

allRatings = Y[np.where(w)]

fig=plt.figure()
plt.hist(allRatings,bins=5)
fig.show()

######################################################
## ten most popular movies:

nRatings = np.sum(w,axis=0)
pop10_ind = np.argsort(nRatings)[-10:][::-1]		## 0 = most popular, 9 = 10th
pop10_text = movies[pop10_ind,1]

fig = plt.figure()
plt.ylabel('popularity')
plt.xlabel('rating')
plt.title('10 most popular movies')

for i in range(10):
	
	ratings = Y[:,pop10_ind[i]]
	ratings = ratings[ratings != 0]
	
	plt.errorbar(x=[np.mean(ratings)],y=[i+1],xerr=[np.std(ratings)],color='black',marker='o')
	plt.text(x=5,y=i+1,s=bytes.decode(pop10_text[i]))
	
fig.show()

######################################################
## ten most highly rated movies:

aveRatings = [ np.mean( Y[:,n][np.where(w[:,n])]) for n in range(Y.shape[1])  ]
aveRatings *= (nRatings >= 10)		## threshold of minimum 10 ratings
best10_ind = np.argsort(aveRatings)[-10:][::-1]		## 0 = most popular, 9 = 10th
best10_text = movies[best10_ind,1]

fig = plt.figure()
plt.ylabel('placement')
plt.xlabel('rating')
plt.title('10 highest rated movies:')

for i in range(10):
	
	ratings = Y[:,best10_ind[i]]
	ratings = ratings[ratings != 0]
	
	plt.errorbar(x=[np.mean(ratings)],y=[i+1],xerr=[np.std(ratings)],color='black',marker='o')
	plt.text(x=5,y=i+1,s=bytes.decode(best10_text[i]))
	
fig.show()


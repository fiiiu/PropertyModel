

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
import time

#sig=0.1

def sigmoid(x, x0, g):
  return 1 / (1 + np.exp(g*(-x-x0)))

def prior(muvec, sigvec):
	return stats.multivariate_normal.pdf(muvec, mean=[0,0,0], cov=2)*\
			stats.invgamma.pdf(sigvec[0], a=1)*stats.invgamma.pdf(sigvec[1], a=1)*\
			stats.invgamma.pdf(sigvec[2], a=1)

	
def slikelihood(choice, ids, muvec, sigvec):
	extmu=np.append(muvec, 0)
	extsig=np.append(sigvec, 1)
	mu=extmu[ids[0]]-extmu[ids[1]]
	sig=extsig[ids[0]]+extsig[ids[1]]
	return 1-(stats.norm.cdf(0, loc=mu, scale=sig)*(-1)**choice+choice)


def likelihood(allchoices, allids, muvec, sigvec):
	lik=1
	for i,choice in enumerate(allchoices):
		ids=allids[i]
		lik*=slikelihood(choice, ids, muvec, sigvec)
	return lik

def posterior(muvec, sigvec, allchoices, allids):
	return likelihood(allchoices, allids, muvec, sigvec)*prior(muvec, sigvec)


def main(datafile='../Data/testdata50.csv'):
	if datafile is None:
		choices = [0,0,0,0,0,0]#0,0,0,0,0]#,0,0,0,1]#,1]
		ids =[[1,0],[1,3],[1,3],[1,3],[3,2],[2,3]]#[0,1], [0,1], [0,1], [2,3], [1,2]]#, [2,3], [1,2], [1,3]]
	else:
		choices=[]
		ids=[]
		with open(datafile) as f:
			for line in f:
				choices.append(int(line[0]))
				ids.append([int(line[2]),int(line[4])])



	npoints=5
	x=np.linspace(-1,1,npoints)
	y=np.linspace(-1,1,npoints)
	z=np.linspace(-1,1,npoints)

	xs=np.linspace(0.01,4,npoints)
	ys=np.linspace(0.01,4,npoints)
	zs=np.linspace(0.01,4,npoints)
	
	#print posterior((1,0,0,0),choices,ids), posterior((1,-1,0,0),choices,ids)
	#post=np.array(map(lambda a,b: np.cos(2*a*b), x, y))
	#post=np.array([[np.cos(2*a*b) for a in x] for b in y])
	post=np.array([[[[[[posterior((a,b,c),(aas,bs,cs),choices,ids) \
		for cs in zs] for bs in ys] for aas in xs]\
		for c in z] for b in y] for a in x])
		
	# post=np.zeros((4,4,4,4,4,4))
	# for i, xi in enumerate(xs):
	# 	for j, yj in enumerate(ys):
	# 		post[0,0,0,0,i,j,0,0]=posterior((1,0,-1,0),(xi,yj,1,1), choices, ids)	
	# print post.shape
	#print np.argmax(post)
	#print post[npoints-1, npoints/2,npoints/2,npoints/2]
	#(xstar,ystar,zstar,wstar)=np.unravel_index(np.argmax(post),post.shape)
	#print (xstar,ystar,zstar,wstar)
	#print x[xstar],y[ystar],z[zstar],w[wstar]
	#print post[xstar,ystar,zstar,wstar]
	#agprint a,b
	#print post.shape
	
	(xstar,ystar,zstar, xxstar, yystar, zzstar)=np.unravel_index(np.argmax(post),post.shape)
	print "Means: {0}".format((x[xstar],y[ystar],z[zstar],0))
	print "Vars: {0}".format((xs[xxstar],ys[yystar],zs[zzstar],1))
	#print x[xstar],y[ystar],z[zstar],w[wstar]
	#print post[xstar,ystar,zstar,wstar]
	
	X,Y=np.meshgrid(x,y)
	#plt.contourf(X,Y, map(lambda b: map(lambda a: 2*a+b, x),y))
	#prig=np.array([[prior((0,0,0,0),(a,b,1,1)) for a in x] for b in y])
	#print prig.shape
	#plt.contourf(X,Y, prig)
	plt.contourf(X,Y,post[:,:,npoints/2,npoints/2,npoints/2,npoints/2])
	#plt.show()

	Xs,Ys=np.meshgrid(xs,ys)
	plt.contourf(Xs,Ys,post[npoints/2,npoints/2,npoints/2,:,:,npoints/2])
	#plt.contourf(Xs,Ys,post[0,0,0,0,:,:,0,0])
	#plt.show()

	with open('post.pkl', 'w') as f:
		pkl.dump(post,f)


if __name__ == '__main__':
	t0=time.time()
	main()
	print "Elapsed: {0:.2f}s".format(time.time()-t0)

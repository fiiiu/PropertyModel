

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

#sig=0.1

def sigmoid(x, x0, g):
  return 1 / (1 + np.exp(g*(-x-x0)))

def prior(muvec, sigvec):
	return stats.multivariate_normal.pdf(muvec, mean=[0,0,0,0], cov=2)*\
			stats.invgamma.pdf(sigvec[0], a=1)*stats.invgamma.pdf(sigvec[1], a=1)*\
			stats.invgamma.pdf(sigvec[2], a=1)*stats.invgamma.pdf(sigvec[3], a=1)

	
def slikelihood(choice, ids, muvec, sigvec):
	#sA=np.random.normal(muvec[ids[0]], sig)
	#sB=np.random.normal(muvec[ids[1]], sig)
	#return int(choice == int(sA>sB)) #Luce here?!
	#return sigmoid((sA-sB)*(-1)**choice,0,2)#, sA-sB)
	mu=muvec[ids[0]]-muvec[ids[1]]
	sig=sigvec[ids[0]]+sigvec[ids[1]]
	#return sigmoid(s*(-1)**choice,0,3)#, sA-sB)
	return 1-(stats.norm.cdf(0, loc=mu, scale=sig)*(-1)**choice+choice)


def likelihood(allchoices, allids, muvec, sigvec):
	lik=1
	for i,choice in enumerate(allchoices):
		ids=allids[i]
		lik*=slikelihood(choice, ids, muvec, sigvec)
	return lik

def posterior(muvec, sigvec, allchoices, allids):
	return likelihood(allchoices, allids, muvec, sigvec)*prior(muvec, sigvec)


def main():
	#print prior([0,0,0,0])
	#print slikelihood(0,[0,1],[0,1,0,0])
	#print likelihood([0,0],[[0,1],[0,1]],[0,1,0,0])
	#print posterior([0,0,0,0], choices, ids)
	
	choices = [0,0,0,0,0]#,0,0,0,1]#,1]
	ids =[[0,1], [0,1], [0,1], [0,1], [1,2]]#, [2,3], [1,2], [1,3]]
	#ids =[[2,3], [2,3], [2,3], [2,3]]#, [1,2]]
	#data = (choices, ids)
	npoints=4
	x=np.linspace(-1,1,npoints)
	y=np.linspace(-1,1,npoints)
	z=np.linspace(-1,1,npoints)
	w=np.linspace(-1,1,npoints)

	xs=np.linspace(0.01,4,npoints)
	ys=np.linspace(0.01,4,npoints)
	zs=np.linspace(0.01,4,npoints)
	ws=np.linspace(0.01,4,npoints)
	
	# post=np.zeros((len(x),len(y)))
	# for i in enumerate(x):
	# 	for j in enumerate(y):
	# 		post[i][j]=np.cos(3*i*j)#posterior([x[i],y[j],0,0], choices, ids)
	
	#print posterior((1,0,0,0),choices,ids), posterior((1,-1,0,0),choices,ids)
	#post=np.array(map(lambda a,b: np.cos(2*a*b), x, y))
	#post=np.array([[np.cos(2*a*b) for a in x] for b in y])
	#post=np.array([[[[[[[[posterior((a,b,c,d),(aas,bs,cs,ds),choices,ids) \
	#	for ds in ws] for cs in zs] for bs in ys] for aas in xs]\
	#	for d in w] for c in z] for b in y] for a in x])
		
	post=np.zeros((4,4,4,4,4,4,4,4))
	for i, xi in enumerate(xs):
		for j, yj in enumerate(ys):
			post[0,0,0,0,i,j,0,0]=posterior((1,0,-1,0),(xi,yj,1,1), choices, ids)	
	print post.shape
	#print np.argmax(post)
	#print post[npoints-1, npoints/2,npoints/2,npoints/2]
	#(xstar,ystar,zstar,wstar)=np.unravel_index(np.argmax(post),post.shape)
	#print (xstar,ystar,zstar,wstar)
	#print x[xstar],y[ystar],z[zstar],w[wstar]
	#print post[xstar,ystar,zstar,wstar]
	#agprint a,b
	#print post.shape
	X,Y=np.meshgrid(x,y)
	#plt.contourf(X,Y, map(lambda b: map(lambda a: 2*a+b, x),y))
	#prig=np.array([[prior((0,0,0,0),(a,b,1,1)) for a in x] for b in y])
	#print prig.shape
	#plt.contourf(X,Y, prig)
	#plt.contourf(X,Y,post[:,:,npoints/2,npoints/2,npoints/2,npoints/2,npoints/2,npoints/2])
	#plt.show()

	Xs,Ys=np.meshgrid(xs,ys)
	#plt.contourf(Xs,Ys,post[npoints/2,npoints/2,npoints/2,npoints/2,:,:,npoints/2,npoints/2])
	plt.contourf(Xs,Ys,post[0,0,0,0,:,:,0,0])
	plt.show()


if __name__ == '__main__':
	main()

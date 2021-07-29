import numpy as np
import pickle
import matplotlib.pyplot as plt
from MDP_RW import MDP_RW

class V_DATA():
	def __init__(self):
		self.V = {}
	def generate(self,V):
		self.V = V
		fp = open("Data/v_data", "wb")
		pickle.dump(self,fp)
		fp.close()
		print("Wrote true V data successfully !")		
														  		
def pi(s,a):
	p = 0.7
	if a == -1 :
		return p
	elif a == 1 or a == 0:
		return 1-p

def iterative_policy_evaluation_sync():
	plot_X = []
	plot_Y = []
	mdp = MDP_RW(11)
	gamma = 1
	V = {}
	V_new = {}	
	for s in range(1,mdp.N+1):
		V[s] = 0
	k = 0
	while True :
		Delta = 0
		for s in range(2,mdp.N+1):
			V_new[s] = 0
			for a in mdp.A[s]:
				V_new[s] += pi(s,a)*(mdp.R(s,a)+gamma*V[mdp.succ(s,a)])
			Delta = max(Delta,abs(V_new[s]-V[s])) 
		for s in range(2,mdp.N+1):
			V[s] = V_new[s]		
		k += 1
		plot_X.append(k)
		plot_Y.append(Delta)
		print(k,Delta)
		if Delta < 1e-3:
			break
	
	
	temp = V_DATA()
	temp.generate(V)
	
	return plot_X,plot_Y 	
			 
def iterative_policy_evaluation_in_place():
	plot_X = []
	plot_Y = []
	mdp = MDP_RW(11)
	gamma = 1
	V = {}
	for s in range(1,mdp.N+1):
		V[s] = 0
	k = 0
	while True:
		Delta = 0
		for s in range(2,mdp.N+1):
			v = 0
			for a in mdp.A[s]:
				v += pi(s,a)*(mdp.R(s,a)+gamma*V[mdp.succ(s,a)])
			Delta = max(Delta,abs(v-V[s]))
			V[s] = v 
		k += 1
		plot_X.append(k)
		plot_Y.append(Delta)
		print(k,Delta)
		if Delta < 1e-3:
			break
	
	'''
	temp = V_DATA()
	temp.generate(V)
	'''
	return plot_X,plot_Y 					

if __name__== "__main__" : 	

	plot_X, plot_Y = iterative_policy_evaluation_sync()
	plt.plot(plot_X,plot_Y)
	plt.xlabel('Iterations, k')
	plt.ylabel('Maximum Absolute Bellman Error, '+r'$\Delta_k$')
	plt.show()

import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from MDP_RW import MDP_RW

gamma = 1
mdp = MDP_RW(11)

class V_DATA():
	def __init__(self):
		self.V = {}

fp = open("Data/v_data", "rb")
temp = pickle.load(fp)
V_true = temp.V
fp.close()
print("Loaded True V data successfully ! ")

class transition():
	def __init__(self,s_t,a_t,r_t_1,s_t_1):
		self.s_t = s_t
		self.a_t = a_t
		self.r_t_1 = r_t_1
		self.s_t_1 = s_t_1	

def Delta(V,V_true,mode):
	if mode == 'MAE':
		delta = -np.Inf
		for s in mdp.S:
			delta = max(delta,abs(V_true[s]-V[s]))
	elif mode == 'RMSE' :
		delta = 0
		N = 0
		for s in mdp.S:
			delta += (V_true[s]-V[s])**2
			N += 1
		delta /= N
		delta = delta**0.5
				
	return delta 
	
def pi(s):
	
	return np.random.choice(np.array(mdp.A[s]), p = np.array([0.7, 0.3]))
	
def BV_TD(s_t,r_t_1,s_t_1,Lambda,alpha,E,V):		

	delta_t = r_t_1 + gamma *V[s_t_1] - V[s_t]
	for s in mdp.S:
		E[s] = E[s] * gamma*Lambda
		if s == s_t :
			E[s] += 1
		V[s] += alpha * delta_t *E[s]						
	
	return E, V
	
def FV_TD(H_episode,Lambda,alpha,V):		

	T = len(H_episode)
	
	for t in range(T):
		s_t = H_episode[t].s_t
		G_t_lambda = 0
		for n in range(1,T+1-t):
			G_t_n = 0
			for t1 in range(t,t+n):
				G_t_n += (H_episode[t1].r_t_1) * gamma**(t1-t)
			G_t_n += (gamma**(n))*V[H_episode[t+n-1].s_t_1]	
			if not(n == T-t):
				G_t_lambda += (1-Lambda)*((Lambda)**(n-1))*G_t_n
			else :
				G_t_lambda += ((Lambda)**(n-1))*G_t_n
		V[s_t] += alpha*(G_t_lambda - V[s_t])	 						

	return V	

if __name__== "__main__" : 
			
	alpha = 0.2
	K = 200
	
	Lambda_List = [0,0.333,0.666,1] 			
	Modes = ['BV','OFFFV']

	plot_Y = {}
	V = {}
	for Lambda in Lambda_List :
		for mode in Modes :
			plot_Y[Lambda,mode] = [1]
			V[Lambda,mode] = {}
			for s in range(1,mdp.N+1):
				if s == 1 :
					V[Lambda,mode][s] = 0
				else :
					V[Lambda,mode][s] = -100			
	
	Delta_0 = Delta(V[Lambda_List[0],'BV'],V_true,'RMSE')
	t_sim = 0
	
	for k in range(1,K+1):
		
		s_t = random.randrange(2,mdp.N+1)
		H_episode = {}
		E = {}
		for Lambda in Lambda_List :
			E[Lambda] = {}
			for s in mdp.S:
				E[Lambda][s] = 0
			H_episode[Lambda] = []
		while True :
			a_t = pi(s_t)
			r_t_1, s_t_1 = mdp.step(s_t,a_t)
			t_sim += 1	
			print(t_sim)			
			for Lambda in Lambda_List :
				E[Lambda], V[Lambda,'BV'] = BV_TD(s_t,r_t_1,s_t_1,Lambda,alpha,E[Lambda],V[Lambda,'BV'])
				plot_Y[Lambda,'BV'].append( Delta(V[Lambda,'BV'],V_true,'RMSE')/Delta_0)
				H_episode[Lambda].append(transition(s_t,a_t,r_t_1,s_t_1))
				#V[Lambda,'ONFV'] = FV_TD(H_episode[Lambda],Lambda,alpha,V[Lambda,'ONFV'])
				#plot_Y[Lambda,'ONFV'].append( Delta(V[Lambda,'ONFV'],V_true,'RMSE')/Delta_0)				
			s_t = s_t_1
			if s_t == 1 :
				break 
			else :
				for Lambda in Lambda_List :
					plot_Y[Lambda,'OFFFV'].append( Delta(V[Lambda,'OFFFV'],V_true,'RMSE')/Delta_0)		
		if s_t == 1:
			for Lambda in Lambda_List :	
				V[Lambda,'OFFFV'] = FV_TD(H_episode[Lambda],Lambda,alpha,V[Lambda,'OFFFV'])
				plot_Y[Lambda,'OFFFV'].append( Delta(V[Lambda,'OFFFV'],V_true,'RMSE')/Delta_0)			

	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
	i = 0
	for Lambda in Lambda_List :
		for mode in Modes :
			if mode == 'BV':
				print_string = 'TD('+str(Lambda)+')'
			elif mode == 'OFFFV':
				print_string = 'Offline '+r'$\lambda$'+'-return ('+str(Lambda)+')'	
				if Lambda == 1 :
					print_string += ' / MC'
			elif mode == 'ONFV':
				print_string = 'Online '+r'$\lambda$'+'-return ('+str(Lambda)+')'	
			plt.plot(np.arange(0,t_sim+1),plot_Y[Lambda,mode],colors[i],label = print_string)
			i += 1
	plt.plot(np.arange(0,t_sim+1),np.zeros(t_sim+1),colors[i])	
	#plt.yticks([0,0.5,1])
	plt.legend(loc=0)
	#plt.title('Model Free Policy Evaluation')
	plt.xlabel('Time steps')
	plt.ylabel('Normalized RMS error')
	plt.show()		
	

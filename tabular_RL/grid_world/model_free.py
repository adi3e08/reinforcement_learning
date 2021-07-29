import numpy as np
import time
import pickle
import random
import matplotlib
import matplotlib.pyplot as plt
########################################

class GRID():
	def __init__(self, H, W, no_mines):
		self.H = H
		self.W = W
		self.no_mines = no_mines
		self.mines = []
		self.start = ()
		self.goal = ()	
	
fp = open("Data/grid_data", "rb")
grid = pickle.load(fp)
fp.close()
print("\n Loaded grid successfully ! \n")		

class Q_DATA():
	def __init__(self):
		self.Q = {}
fp = open("Data/q_data", "rb")
temp = pickle.load(fp)
Q_true = temp.Q
fp.close()
print("Loaded True Q data successfully ! ")					
#################
class MDP():
	def __init__(self):
		print("--------------------------------------------------------------------\n")	
		self.A_tot = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
		self.dimA = 8
		self.S= []
		self.A = {}
		for h in range(grid.H):
			for w in range(grid.W):
				if((h,w) not in grid.mines):
					self.S.append((h,w))
		for s in self.S:
			if( s == grid.goal ):
				self.A[s] = []
			else :		
				A_s = []
				for a in self.A_tot:
					s1 = (s[0] + a[0],s[1] + a[1]) 
					if s1 in self.S :
						A_s.append(a)
				self.A[s] = A_s
		if not(self.A[grid.start]):
			print("Start surrounded by obstacles ! ")
					
	def R(self,s,a):
		
		if abs(a[0])== 1 and abs(a[1])==1:
			return - 2**0.5
		else :
			return -1
		
		#return -1	
	
	def succ(self,s,a):
		return (s[0] + a[0],s[1] + a[1])	
		
	def pred(self,s):
		pred_s = []
		for s1 in mdp.S:
			for a in mdp.A[s1]:
				if(self.succ(s1,a) == s):
					pred_s.append(s1)
					break	
		return pred_s			
	
	def step(self,s,a):
		return self.R(s,a),self.succ(s,a)		
														  		
mdp = MDP()
gamma = 1-1e-5
max_episode_steps = max(grid.H,grid.W)*(2)
			
def epsilon_greedy(Q,s_t,epsilon):
	max_Q_s_t_a = -np.Inf
	max_index = 0
	i = 0
	for a in mdp.A[s_t]:
		if Q[s_t,a] > max_Q_s_t_a :
			max_Q_s_t_a = Q[s_t,a]
			max_index = i
		i += 1	
	if epsilon :
		dim_A_s_t = len(mdp.A[s_t])
		p1 = (epsilon/dim_A_s_t)*np.ones(dim_A_s_t)
		p1[max_index] += (1-epsilon)
		#p1 = ((1-epsilon)/(dim_A_s_t-1))*np.ones(dim_A_s_t)
		#p1[max_index] = epsilon
		a_index = np.random.choice(np.arange(dim_A_s_t), p = p1)
	else :
		a_index = max_index
		
	a_t = mdp.A[s_t][a_index]				

	return a_t
					
def print_path_Q(Q):
	
	fig = plt.figure()
	ax = fig.gca()			
	ax.set_xticks(np.arange(0, grid.H , 1))
	ax.set_yticks(np.arange(0, grid.W , 1))		
	if grid.no_mines :
		M = np.zeros((grid.no_mines,2))			
		for i in range(grid.no_mines):
			M[i,0] = grid.mines[i][0]
			M[i,1] = grid.mines[i][1] 
		plt.plot(M[:,0], M[:,1], 'rx',label = 'Obstacle')
	plt.plot(grid.start[0],grid.start[1],'g^',label = 'Start')
	plt.plot(grid.goal[0],grid.goal[1],'gv',label = 'Goal')
		
	s_t = grid.start
	for t in range(max_episode_steps):
		a_t = epsilon_greedy(Q,s_t,0)
		s_t_1 = mdp.succ(s_t,a_t)
		plt.arrow(s_t[0],s_t[1],s_t_1[0]-s_t[0],s_t_1[1]-s_t[1],fc="k", ec="k",head_width=0.2, head_length=0.1 )
		s_t = s_t_1
		print(t)
		if s_t == grid.goal :
			break
	plt.grid(True)
	plt.show()	

def Delta(Q,mode):
	if mode == 'MAE':
		delta = -np.Inf
		for s in mdp.S:
			for a in mdp.A[s]:
				delta = max(delta,abs(Q_true[s,a]-Q[s,a]))
	elif mode == 'RMSE' :
		delta = 0
		N = 0
		for s in mdp.S:
			for a in mdp.A[s]:
				delta += (Q_true[s,a]-Q[s,a])**2
				N += 1
		delta /= N
		delta = delta**0.5
				
	return delta 			 		

def SARSA_Lambda(Lambda):		

	alpha = 1.0
	K = 1500

	plot_X = []
	plot_Y = []

	Q = {}
	E = {}
		
	for s in mdp.S:
		for a in mdp.A[s]:
			Q[s,a] = -100

	k = 0
	error = Delta(Q,'RMSE')
	print("Episode : ",k," Delta : ",error)	
	plot_X.append(k)
	plot_Y.append(error)
	
	for k in range(1,K+1):
		
		if k <= 1000:
			epsilon = 1 - 0.9999 / 999 * (k-1)			
		else :
			epsilon = 0.0
		
		
		#epsilon = 0.0
		while True :
			s_t = mdp.S[random.randrange(len(mdp.S))]
			if not(s_t == grid.goal) :
				a_t = mdp.A[s_t][random.randrange(len(mdp.A[s_t]))]
				break
		
		total_reward = 0
		
		for s in mdp.S:
			for a in mdp.A[s]:
				E[s,a] = 0
		
		for t in range(max_episode_steps):
			r_t_1, s_t_1 = mdp.step(s_t,a_t)
			total_reward += r_t_1
			if s_t_1 == grid.goal : 
				Q_s_t_a_t_target = r_t_1
			else :	
				a_t_1 = epsilon_greedy(Q,s_t_1,epsilon)
				Q_s_t_a_t_target = r_t_1 + gamma*Q[s_t_1,a_t_1]								 
			delta_t =  Q_s_t_a_t_target - Q[s_t,a_t]
			E[s_t,a_t] += 1
			for s in mdp.S:
				for a in mdp.A[s]:
					Q[s,a] += alpha * delta_t *E[s,a]
					E[s,a] = E[s,a] * gamma*Lambda		
			if s_t_1 == grid.goal :
				break 
			else :
				s_t = s_t_1
				a_t = a_t_1
					
		error = Delta(Q,'RMSE')
		print("Episode : ",k," Delta : ",error)	
		plot_X.append(k)
		plot_Y.append(error)
	
	#print_path_Q(Q)	
	
	return plot_X, plot_Y

def Q_Lambda(Lambda):		
	alpha = 1.0
	K = 1500
	plot_X = []
	plot_Y = []
	Q = {}
	E = {}
	N = {}
		
	for s in mdp.S:
		for a in mdp.A[s]:
			Q[s,a] = -1000
	k = 0
	error = Delta(Q,'RMSE')
	print("Episode : ",k," Delta : ",error)	
	plot_X.append(k)
	plot_Y.append(error)
	
	for k in range(1,K+1):
		
		if k <= 1000:
			epsilon = 1 - 0.9999 / 999 * (k-1)			
		else :
			epsilon = 0.0
		
		
		#epsilon = 0.0
		while True :
			s_t = mdp.S[random.randrange(len(mdp.S))]
			if not(s_t == grid.goal) and mdp.A[s_t]:
				break
		
		a_t = epsilon_greedy(Q,s_t,epsilon)
		
		total_reward = 0
		
		for s in mdp.S:
			for a in mdp.A[s]:
				E[s,a] = 0
		
		for t in range(max_episode_steps):
			r_t_1, s_t_1 = mdp.step(s_t,a_t)
			total_reward += r_t_1
			if s_t_1 == grid.goal : 
				Q_s_t_a_t_target = r_t_1
			else :	
				a_t_1 = epsilon_greedy(Q,s_t_1,epsilon)
				a_t_1_max = epsilon_greedy(Q,s_t_1,0)
				if Q[s_t_1,a_t_1] == Q[s_t_1,a_t_1_max]:
					a_t_1_max = a_t_1
				Q_s_t_a_t_target = r_t_1 + gamma*Q[s_t_1,a_t_1_max]								 
			delta_t =  Q_s_t_a_t_target - Q[s_t,a_t]
			E[s_t,a_t] += 1
			for s in mdp.S:
				for a in mdp.A[s]:
					Q[s,a] += alpha * delta_t *E[s,a]
					if not(s_t_1 == grid.goal) :
						if a_t_1_max == a_t_1 :
							E[s,a] = E[s,a] * gamma*Lambda
						else :
							E[s,a] = 0			
			s_t = s_t_1
			if s_t == grid.goal :
				break 
			else :
				a_t = a_t_1	
		error = Delta(Q,'RMSE')
		print("Episode : ",k," Delta : ",error)	
		plot_X.append(k)
		plot_Y.append(error)
	
	#print_path_Q(Q)	
	
	return plot_X, plot_Y


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

Lambda = [0, 0.2, 0.4, 0.6, 0.8, 1] 			
for i in range(6):
	print_string = 'Q('+str(Lambda[i])+')'
	print(print_string)
	#plot_X, plot_Y = SARSA_Lambda(Lambda[i])
	plot_X, plot_Y = Q_Lambda(Lambda[i])
	plt.plot(plot_X,plot_Y,colors[i],label = print_string)
	#plot_X, plot_Y = Q_Lambda(Lambda[i])
	#plt.plot(plot_X,plot_Y,colors[i+4],label = 'Q('+str(Lambda[i])+')')
plt.plot(plot_X,np.zeros(len(plot_X)),colors[6],label = 'Delta = 0')	
plt.legend(loc=0)
plt.xlabel('Iterations, k')
plt.ylabel('Delta (k) ')
plt.show()		

'''
Lambda = 0.666
#plot_X, plot_Y = SARSA_Lambda(Lambda)
#plt.plot(plot_X,plot_Y,colors[0],label = 'SARSA('+str(Lambda)+')')
plot_X, plot_Y = Q_Lambda(Lambda)
plt.plot(plot_X,plot_Y,colors[1],label = 'Q('+str(Lambda)+')')
plt.plot(plot_X,np.zeros(len(plot_X)),colors[2],label = 'Delta = 0')
plt.legend(loc=0)
plt.xlabel('Iterations, k')
plt.ylabel('Delta (k) ')
plt.show()		
'''

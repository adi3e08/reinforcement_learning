import numpy as np
import time
import pickle
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
		
		return -(a[0]**2+a[1]**2)**0.5
		
	def succ(self,s,a):
		return (s[0] + a[0],s[1] + a[1])	
		
	def pred(self,s):
		pred_s = []
		for s1 in mdp.S:
			if not(s1==grid.goal):
				for a in mdp.A[s1]:
					if(self.succ(s1,a) == s) :
						pred_s.append(s1)
						break	
		return pred_s			
	
	def step(self,s,a):
		return self.R(s,a),self.succ(s,a)			
														  		
mdp = MDP()
gamma = 1
		 	
def greedy(V,s,mode):
	
	max_Q_s_a = -np.Inf
	a_best_s = ()
	for a in mdp.A[s]:
		Q_s_a = mdp.R(s,a)+gamma*V[mdp.succ(s,a)] 
		if Q_s_a > max_Q_s_a :
			max_Q_s_a = Q_s_a
			a_best_s = a
	
	if mode == 'Q':
		return max_Q_s_a
	elif mode == 'a' :
		return a_best_s	 
				
def print_path_V(V):
	max_episode_steps = max(grid.H,grid.W)**2
	fig = plt.figure()
	ax = fig.gca()			
	ax.set_xticks(np.arange(0, grid.H+1 , 1))
	ax.set_yticks(np.arange(0, grid.W+1 , 1))		
	if grid.no_mines :
		M = np.zeros((grid.no_mines,2))			
		for i in range(grid.no_mines):
			M[i,0] = grid.mines[i][0]+0.5
			M[i,1] = grid.mines[i][1]+0.5
		plt.plot(M[:,0], M[:,1], 'rx',label = 'Obstacle')
	plt.plot(grid.start[0]+0.5,grid.start[1]+0.5,'g^',label = 'Start')
	plt.plot(grid.goal[0]+0.5,grid.goal[1]+0.5,'gv',label = 'Goal')
		
	s_t = grid.start
	for t in range(max_episode_steps):
		a_t = greedy(V,s_t,'a')
		s_t_1 = mdp.succ(s_t,a_t)
		plt.arrow(s_t[0]+0.5,s_t[1]+0.5,s_t_1[0]-s_t[0],s_t_1[1]-s_t[1],fc="k", ec="k",head_width=0.2, head_length=0.1 )
		s_t = s_t_1
		print(t)
		if s_t == grid.goal :
			break
	plt.grid(True)
	plt.show()	

class Q_DATA():
	def __init__(self):
		self.Q = {}
	def generate(self,V):
		for s in mdp.S:
			for a in mdp.A[s]:
				self.Q[s,a] = mdp.R(s,a)+gamma*V[mdp.succ(s,a)]
		fp = open("Data/q_data", "wb")
		pickle.dump(self,fp)
		fp.close()
		print("Wrote true Q data successfully !")					
		  
def Value_Iteration_Sync():
	t_start = time.time()
	V = {}
	V_old = {}
	plot_X = []
	plot_Y = []
	for s in mdp.S:
		if( s == grid.goal ):
			V_old[s] = 0
		else :		
			V_old[s] = -100		
	k = 0

	while True:
		k += 1
		delta_max = -np.Inf
		for s in mdp.S :
			if mdp.A[s]:
				V[s] = greedy(V_old,s,'Q')
			else :
				V[s] = V_old[s]
			delta_s = abs(V[s]-V_old[s])
			if delta_s > delta_max :
				delta_max = delta_s
		print(k,delta_max)
		plot_X.append(k)
		plot_Y.append(delta_max)
		for s in mdp.S :
			V_old[s] = V[s]
		if delta_max < 1e-5:
			break					 	
	t_end = time.time()
	print("Time taken : ",t_end - t_start)
	#temp = Q_DATA()
	#temp.generate(V)
	
	'''
	plt.plot(plot_X,plot_Y)
	plt.xlabel('Iterations, k')
	plt.ylabel(r'$\Delta_{max}(k)$')
	plt.show()
	'''
	print_path_V(V)
	
	return plot_X,plot_Y
	
def Value_Iteration_In_Place():
	t_start = time.time()
	V = {}
	plot_X = []
	plot_Y = []
	for s in mdp.S:
		if( s == grid.goal ):
			V[s] = 0
		else :		
			V[s] = -100		
	k = 0

	while True:
		k += 1
		delta_max = -np.Inf
		for s in mdp.S :
			V_old_s = V[s]
			if mdp.A[s]:
				V[s] = greedy(V,s,'Q')
			delta_s = abs(V[s]-V_old_s)
			if delta_s > delta_max :
				delta_max = delta_s
		print(k,delta_max)
		plot_X.append(k)
		plot_Y.append(delta_max)
		if delta_max < 1e-5:
			break					 	
	t_end = time.time()
	print("Time taken : ",t_end - t_start)	
	#temp = Q_DATA()
	#temp.generate(V)	
	'''
	plt.plot(plot_X,plot_Y)
	plt.xlabel('Iterations, k')
	plt.ylabel(r'$\Delta_{max}(k)$')
	plt.show()
	print_path_V(V)
	'''	
	return plot_X,plot_Y
	
class PQ():
	def __init__(self):
		self.queue= []
	
	def swap(self,i,j):
		c = self.queue[i]
		self.queue[i] = self.queue[j]
		self.queue[j] = c
		
	def insert(self,s,delta_s):
		self.queue.append((s,delta_s))
		if len(self.queue) > 1 :
			for i in range(len(self.queue)-1,0,-1):
				if self.queue[i][1] > self.queue[i-1][1]:
					self.swap(i,i-1)
				else : 
					break
	def append(self,s,delta_s):
		self.queue.append((s,delta_s))		
	
	def delete(self,s):
		for i in range(len(self.queue)):
			if self.queue[i][0] == s :
				del self.queue[i]
				break			
	
	def max(self):
		states = []
		for i in range(len(self.queue)):
			if self.queue[i][1] == self.queue[0][1]:
				states.append(self.queue[i][0])
			else :
				break
		return states		 			
						
			 		
def Value_Iteration_ASync(mode):
	t_start = time.time()
	V = {}
	pq = PQ()
	plot_X = []
	plot_Y = []

	for s in mdp.S:
		if( s == grid.goal ):
			V[s] = 0
		else :		
			V[s] = -100	
	for s in mdp.S:
		if not(s == grid.goal):
			pq.insert(s,abs(greedy(V,s,'Q')-V[s]))
	
	k = 0
	#print(k,pq.queue[0][1])
	plot_X.append(k)
	plot_Y.append(pq.queue[0][1])
	
	while pq.queue[0][1] > 1e-3:
		k += 1
		V_new = {}
		states = pq.max()
		if mode == 0:
			for s in states:
				V_new[s] = greedy(V,s,'Q')
			for s in states:
				V[s] = V_new[s]
				pq.delete(s)
				pq.append(s,0)			
		else :
			for s in states:
				V[s] = greedy(V,s,'Q')
				pq.delete(s)
				pq.append(s,0)			
		
		for s in states:
			for s1 in mdp.pred(s): 
				if not(s1 in states):
					pq.delete(s1)
					pq.insert(s1,abs(greedy(V,s1,'Q')-V[s1]))
		#print(k,pq.queue[0][1])
		plot_X.append(k)
		plot_Y.append(pq.queue[0][1])		
		
	
	t_end = time.time()
	print("Time taken : ",t_end - t_start)			 	
	
	fig = plt.figure()
	plt.plot(plot_X,plot_Y)
	plt.xlabel('Iterations, k')
	plt.ylabel(r'$\Delta_{max}(k)$')
	plt.show()

	#print_path_V(V)					 		

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
	
plot_X1, plot_Y1 = Value_Iteration_Sync()
plt.plot(plot_X1,plot_Y1,colors[0],label = 'Synchronous Updates')
plot_X2, plot_Y2 = Value_Iteration_In_Place()
plt.plot(plot_X2,plot_Y2,colors[1],label = 'In-Place Updates')
plt.legend(loc=0)
#plt.title('Model Based Policy Evaluation')
plt.xlabel('Iterations')
#plt.ylabel(r'$\Delta_k$')
plt.ylabel(r'$\Delta$')
plt.show()				

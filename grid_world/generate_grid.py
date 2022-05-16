import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

class GRID():
	
	def __init__(self, H, W, no_mines):
		
		self.H = H
		self.W = W
		self.no_mines = no_mines
		self.mines = []
		self.start = ()
		self.goal = ()	
		
	def generate(self):
		
		self.start = (self.H-1,0)
		self.goal = (0,self.W-1)
		
		for i in range(self.no_mines):
			while True:
				temp = random.sample(range(self.H*self.W),1)
				s = (temp[0]//self.W,temp[0] % self.W)
				if not(s== self.start or s==self.goal): 
					self.mines.append(s)
					break
			print(i)		
				
		fp = open("Data/grid_data", "wb")
		pickle.dump(self,fp)
		fp.close()
		print("Wrote grid data successfully !")						
	
	def print_grid(self):

		fig = plt.figure()
		ax = fig.gca()			
		ax.set_xticks(np.arange(0, self.H+1 , 1))
		ax.set_yticks(np.arange(0, self.W+1 , 1))
		if self.no_mines :
			M = np.zeros((self.no_mines,2))			
			for i in range(self.no_mines):
				M[i,0] = self.mines[i][0]+0.5
				M[i,1] = self.mines[i][1]+0.5 
			plt.plot(M[:,0], M[:,1], 'rx')
		plt.plot(self.start[0]+0.5,self.start[1]+0.5,'g^')
		plt.plot(self.goal[0]+0.5,self.goal[1]+0.5,'gv')
		plt.grid()
		#plt.legend()
		plt.show()	

'''
grid = GRID(40,40,800)
grid.generate()
'''
fp = open("Data/grid_data", "rb")
grid = pickle.load(fp)
fp.close()
print(grid.no_mines)
print("\n Loaded grid successfully ! \n")
grid.print_grid()
	


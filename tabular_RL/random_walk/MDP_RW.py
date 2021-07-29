#################
class MDP_RW():
	def __init__(self,N):
		self.N = N
		self.S= []
		self.A = {}
		for s in range(1,self.N+1):
			self.S.append(s)
		for s in self.S:
			if  s == 1 :
				self.A[s] = []
			elif s == self.N :
				self.A[s] = [-1,0]
			else :		
				self.A[s] = [-1,1]

	def R(self,s,a):
		
		return 1
				
	def succ(self,s,a):
		return s + a	
		
	def step(self,s,a):
		return self.R(s,a),self.succ(s,a)	

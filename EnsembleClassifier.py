import numpy as np


class EnsembleClassifier():


	def __init__(self, type):
		self.type = type


	def predict1(self, B_op, E_op, N_op):
		final_op = {}
		final_voter = {}
		qIds = list(B_op.keys())
		for qId in qIds:
			votes = [0,0,0]
			if (qId in B_op):
				votes[0] +=1
			if (qId in E_op):
				votes[1] +=1
			if (qId in N_op):
				votes[2] +=1
			if ((qId in B_op) and (qId in E_op) and (B_op[qId]==E_op[qId])): 
				votes[0] +=1
				votes[1] +=1
			if ((qId in B_op) and (qId in N_op) and (B_op[qId]==N_op[qId])): 
				votes[0] +=1
				votes[2] +=1
			if ((qId in E_op) and (qId in N_op) and (E_op[qId]==N_op[qId])): 
				votes[1] +=1
				votes[2] +=1
			votesmaxindex = np.argmax(votes)
			final_op[qId] = B_op[qId] if votesmaxindex==0 else (E_op[qId] if votesmaxindex==1 else N_op[qId])
			final_voter[qId] =  { "maxVotes": "B" if votesmaxindex==0 else ("E" if votesmaxindex==1 else "N"),
									"votes": votes }
		return final_op, final_voter


	def predict2(self, B_op, E_op, N_op):
		final_op = {}
		final_voter = {}
		qIds = list(B_op.keys())
		for qId in qIds:
			votes = [0,0,0]
			isNonAnswerable = False
			temp_cnt = 0
			if (qId in B_op):
				votes[0] +=1
			if (qId in E_op):
				votes[1] +=1
			if (qId in N_op):
				votes[2] +=1
			if ((qId in B_op) and (qId in E_op) and (B_op[qId]==E_op[qId])): 
				votes[0] +=1
				votes[1] +=1
			if ((qId in B_op) and (qId in N_op) and (B_op[qId]==N_op[qId])): 
				votes[0] +=1
				votes[2] +=1
			if ((qId in E_op) and (qId in N_op) and (E_op[qId]==N_op[qId])): 
				votes[1] +=1
				votes[2] +=1
			votesmaxindex = np.argmax(votes)
			final_op[qId] = B_op[qId] if votesmaxindex==0 else (E_op[qId] if votesmaxindex==1 else N_op[qId])
			if (final_op[qId]!=""):
				final_op[qId] = B_op[qId]
				final_voter[qId] =  { "maxVotes": "B", "votes": votes }
			else:
				final_voter[qId] =  { "maxVotes": "B" if votesmaxindex==0 else ("E" if votesmaxindex==1 else "N"),
									"votes": votes }
		return final_op, final_voter
import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
	'''
	base class for model selection (strategy design pattern)
	'''

	def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
				 n_constant=3,
				 min_n_components=2, max_n_components=10,
				 random_state=14, verbose=False):
		self.words = all_word_sequences
		self.hwords = all_word_Xlengths
		self.sequences = all_word_sequences[this_word] # dict, EXAMPLE : 'PUTASIDE': [[[2, 68, 21, 139], [6, 58, 23, 151], [11, 57, 29, 165], [21, 58, 32, 176],....
		self.X, self.lengths = all_word_Xlengths[this_word]
		self.this_word = this_word
		self.n_constant = n_constant
		self.min_n_components = min_n_components
		self.max_n_components = max_n_components
		self.random_state = random_state
		self.verbose = verbose

	def select(self):
		raise NotImplementedError

	def base_model(self, num_states):
		# with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		# warnings.filterwarnings("ignore", category=RuntimeWarning)
		try:
			hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
									random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
			if self.verbose:
				print("model created for {} with {} states".format(self.this_word, num_states))
			return hmm_model
		except:
			if self.verbose:
				print("failure on {} with {} states".format(self.this_word, num_states))
			return None


class SelectorConstant(ModelSelector):
	""" select the model with value self.n_constant

	"""

	def select(self):
		""" select based on n_constant value

		:return: GaussianHMM object
		"""
		best_num_components = self.n_constant
		return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
	""" select the model with the lowest Bayesian Information Criterion(BIC) score

	http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
	Bayesian information criteria: BIC = -2 * logL + p * logN
	"""

	def select(self):
		""" select the best model for self.this_word based on
		BIC score for n between self.min_n_components and self.max_n_components

		:return: GaussianHMM object
		"""
		warnings.filterwarnings("ignore", category=DeprecationWarning)
	   
		# Init best_score
		best_score = float('inf')

		# Rows (number of datapoints)
		logN = np.log(self.X.shape[0])

		# Cols (length per datapoint)
		n_features = self.X.shape[1]			

		# for each hidden state...
		for n in range(self.min_n_components, self.max_n_components + 1):
			try:
				model = self.base_model(n)
				logL = model.score(self.X, self.lengths)

				# number of parameters calc
				n_params = n * (n - 1) + 2 * n_features * n
				
				# BIC = -2 * logL + p * logN
				bic_score = -2 * logL + n_params * logN
			except:
				bic_score = float('inf')

			# Keep the best score		
			if bic_score < best_score:
				best_score = bic_score
				best_model = model

		best_model = self.base_model(self.n_constant) if best_score == float('inf') else best_model

		return best_model


class SelectorDIC(ModelSelector):
	''' select best model based on Discriminative Information Criterion

	Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
	Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
	https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
	DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
	'''

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		
		best_dic = float("-inf")
		dic_best_n_states = self.n_constant

		# Create a list of sequences of words -to calculate antilikehood- and 
		other_words = list(self.words)

		#remove the current word from the list to calculate the anti-likelihood
		other_words.remove(self.this_word)
		
		# for each hidden state...
		for n in range(self.min_n_components, self.max_n_components+1):
			try:
				# Fit model 
				model = self.base_model(n)

				# Get log-likelihood 
				logL = model.score(self.X, self.lengths)

				other_scores = 0.0

				# Get low scores for the remaining words 
				for word in other_words:

					# X, lengths 
					X, lengths = self.hwords[word]

					# Sum scores
					other_scores += model.score(X, lengths)
		
				# Calculate DIC 
				dic =  logL - other_scores /(len(self.words) - 1)

				# Keep best score
				if best_dic < dic:
					best_dic = dic
					dic_best_n_states = n
			except:
					pass

		return self.base_model(dic_best_n_states)


class SelectorCV(ModelSelector):
	''' select best model based on average log Likelihood of cross-validation folds

	'''

	def select(self):
	   
		warnings.filterwarnings("ignore", category=DeprecationWarning)

		best_score = float('-inf') # init score
		best_n_states = self.n_constant 		 

		if len(self.sequences) > 1: # split if 2 or more word sequences
			folds = KFold(shuffle=True, n_splits=min(len(self.sequences),3))
		
		else: # otherwise return base model
			return self.base_model(self.n_constant)
				
		for n in range(self.min_n_components, self.max_n_components + 1): # for n number of hidden states...
			
			fold_scores = []

			for cv_train_idx, cv_test_idx in folds.split(self.sequences):
				
				# Training data
				self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
				
				# Testing data
				X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

				#  wrap model.score() into try/except, because not all models are solvable	
				try:
					# Fit model
					model = GaussianHMM(n_components=n_state, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
					
					# get log Likelihood
					logL = model.score(X_test, lengths_test)

					# Get scores
					fold_scores.append(logL)
				
				except:
					continue

			# Metrics is based on average scores       
			score = np.mean(fold_scores) if len(fold_scores) > 0 else float('-inf')
			
			# Keep the best score
			if score > best_score:
				best_score = score
				best_num_states = n 
   
		# Return trained model with best number of states
		return self.base_model(best_n_states)


# coding: utf-8
"""CS585: Assignment 2

In this assignment, you will complete an implementation of
a Hidden Markov Model and use it to fit a part-of-speech tagger.
"""

from collections import Counter, defaultdict
import math
import numpy as np
import os.path
import urllib.request


class HMM:
	def __init__(self, smoothing=0):
		"""
		Construct an HMM model with add-k smoothing.
		Params:
		  smoothing...the add-k smoothing value
		
		This is DONE.
		"""
		self.smoothing = smoothing

	def fit_transition_probas(self, tags):
		"""
		Estimate the HMM state transition probabilities from the provided data.

		Creates a new instance variable called `transition_probas` that is a 
		dict from a string ('state') to a dict from string to float. E.g.
		{'N': {'N': .1, 'V': .7, 'D': 2},
		 'V': {'N': .3, 'V': .5, 'D': 2},
		 ...
		}
		See test_hmm_fit_transition.
		
		Params:
		  tags...a list of lists of strings representing the tags for one sentence.
		Returns:
			None
		"""
		self.states = sorted(np.unique([state for t in tags for state in t]))
		bigrams = []
		all_states = []
		for t in tags:
			bigrams.extend(b for b in zip(t[:-1], t[1:]))
			all_states.extend(t[:-1])

		u_states = np.unique(all_states)
		state_counts = (Counter(all_states))
		bigram_counts = (Counter(bigrams))

		self.transition_probas = {}
		for s1 in self.states:
			state_probs = {}
			for s2 in self.states:
				numerator = (s1, s2)
				denominator = s1
				prob = (bigram_counts[numerator] + self.smoothing) / (
					state_counts[denominator] + (len(u_states) * self.smoothing))
				state_probs[s2] = prob
			self.transition_probas[s1] = state_probs

	def fit_emission_probas(self, sentences, tags):
		"""
		Estimate the HMM emission probabilities from the provided data. 

		Creates a new instance variable called `emission_probas` that is a 
		dict from a string ('state') to a dict from string to float. E.g.
		{'N': {'dog': .1, 'cat': .7, 'mouse': 2},
		 'V': {'run': .3, 'go': .5, 'jump': 2},
		 ...
		}

		Params:
		  sentences...a list of lists of strings, representing the tokens in each sentence.
		  tags........a list of lists of strings, representing the tags for one sentence.
		Returns:
			None		  

		See test_hmm_fit_emission.
		"""
		match_tuples = []
		all_states = []
		all_words = []
		for i in range(0, len(tags)):
			match_tuples.extend(m for m in zip(sentences[i], tags[i]))
			all_states.extend(tags[i])
			all_words.extend(sentences[i])

		words = np.unique(all_words)
		state_counts = (Counter(all_states))
		match_counts = (Counter(match_tuples))

		self.emission_probas = {}
		for s1 in all_states:
			state_probs = {}
			for s2 in all_words:
				numerator = (s2, s1)
				denominator = s1
				prob = (match_counts[numerator] + self.smoothing) / (
				state_counts[denominator] + (len(words) * self.smoothing))
				state_probs[s2] = prob
			self.emission_probas[s1] = state_probs

	def fit_start_probas(self, tags):
		"""
		Estimate the HMM start probabilities form the provided data.

		Creates a new instance variable called `start_probas` that is a 
		dict from string (state) to float indicating the probability of that
		state starting a sentence. E.g.:
		{
			'N': .4,
			'D': .5,
			'V': .1		
		}

		Params:
		  tags...a list of lists of strings representing the tags for one sentence.
		Returns:
			None

		See test_hmm_fit_start
		"""
		self.start_probas = {}
		start_labels = [t[0] for t in tags]
		start_counts = Counter(start_labels)
		for s in self.states:
			self.start_probas[s] = (start_counts[s] + self.smoothing) / (
			len(tags) + (len(self.states) * self.smoothing))

	def fit(self, sentences, tags):
		"""
		Fit the parameters of this HMM from the provided data.

		Params:
		  sentences...a list of lists of strings, representing the tokens in each sentence.
		  tags........a list of lists of strings, representing the tags for one sentence.
		Returns:
			None		  

		DONE. This just calls the three fit_ methods above.
		"""
		self.fit_transition_probas(tags)
		self.fit_emission_probas(sentences, tags)
		self.fit_start_probas(tags)


	def viterbi(self, sentence):
		"""
		Perform Viterbi search to identify the most probable set of hidden states for
		the provided input sentence.

		Params:
		  sentence...a lists of strings, representing the tokens in a single sentence.

		Returns:
		  path....a list of strings indicating the most probable path of POS tags for
		  		  this sentence.
		  proba...a float indicating the probability of this path.
		"""
		rows = len(self.states)
		cols = len(sentence)
		V = np.empty([rows, cols])
		B = np.empty([rows, cols])

		for i in range(0, len(self.states)):
			state = self.states[i]
			word = sentence[0]
			V[i, 0] = self.start_probas[state] * self.emission_probas[state][word]
			B[i, 0] = 0

		for t in range(1, len(sentence)):
			word = sentence[t]
			for s in range(0, len(self.states)):
				s1 = self.states[s]
				vals = []
				back_ptrs = []
				for sp in range(0, len(self.states)):
					s2 = self.states[sp]
					vt = V[sp, t - 1] * self.transition_probas[s2][s1] * self.emission_probas[s1][word]
					vals.append(vt)
					bp = V[sp, t - 1] * self.transition_probas[s2][s1]
					back_ptrs.append(bp)
				V[s, t] = max(vals)
				B[s, t] = np.argmax(back_ptrs)
		vf = max(V[:, t])
		bf = np.argmax(V[:, t])

		state_idx = []
		state_idx.append(bf)
		b_pt = bf

		for t in range(len(sentence) - 1, 0, -1):
			b_pt = int(B[b_pt, t])
			state_idx.append(b_pt)

		path = []
		for idx in state_idx[::-1]:
			path.append(self.states[idx])

		return path, vf


def read_labeled_data(filename):
	"""
	Read in the training data, consisting of sentences and their POS tags.

	Each line has the format:
	<token> <tag>

	New sentences are indicated by a newline. E.g. two sentences may look like this:
	<token1> <tag1>
	<token2> <tag2>

	<token1> <tag1>
	<token2> <tag2>
	...

	See data.txt for example data.

	Params:
	  filename...a string storing the path to the labeled data file.
	Returns:
	  sentences...a list of lists of strings, representing the tokens in each sentence.
	  tags........a lists of lists of strings, representing the POS tags for each sentence.
	"""
	sentences = []
	tags = []

	with open(fname, 'r') as fop:
		s = []
		t = []
		for line in fop:
			if line == '\n':
				sentences.append(s)
				tags.append(t)
				s = []
				t = []

			else:
				values = line.split()
				s.append(values[0].strip())
				t.append(values[1].strip())
	return sentences, tags

def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/ty7cclxiob3ajog/data.txt?dl=1'
    urllib.request.urlretrieve(url, 'data.txt')

if __name__ == '__main__':
	"""
	Read the labeled data, fit an HMM, and predict the POS tags for the sentence
	'Look at what happened'

	DONE - please do not modify this method.

	The expected output is below. (Note that the probability may differ slightly due
	to different computing environments.)

	$ python3 a2.py  
	model has 34 states
        ['$', "''", ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '``']
	predicted parts of speech for the sentence ['Look', 'at', 'what', 'happened']
	(['VB', 'IN', 'WP', 'VBD'], 2.751820088075314e-10)
	"""
	fname = 'data.txt'
	if not os.path.isfile(fname):
		download_data()
	sentences, tags = read_labeled_data(fname)

	model = HMM(.001)
	model.fit(sentences, tags)
	print('model has %d states' % len(model.states))
	print(model.states)
	sentence = ['Look', 'at', 'what', 'happened']
	print('predicted parts of speech for the sentence %s' % str(sentence))
	print(model.viterbi(sentence))

# coding: utf-8
"""CS585: Assignment 3

In this assignment, you will build a named-entity classifier
using LogisticRegression.

We'll use the labeled data from the CoNLL 2003 Shared Task:
http://www.cnts.ua.ac.be/conll2003/ner/

This is downloaded by the download_data method below.

The main goals of this assignment are to have you:
1- Implement different feature sets for the classifier.
2- Compute evaluation metrics for the classifier on the test set.
3- Enumerate over various settings of the features to determine
   which features result in the highest accuracy.

See Log.txt for the expected output of running the main method.
(Subject to minor variants based on computing environment.)
"""

### DO NOT ADD TO THESE IMPORTS. ####
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.request
#####################################


def download_data():
	""" Download labeled data.
	DONE ALREADY.
	"""
	url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
	urllib.request.urlretrieve(url, 'train.txt')
	url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'
	urllib.request.urlretrieve(url, 'test.txt')


def read_data(filename):
	"""
	Read the data file into a list of lists of tuples.

	Each sentence is a list of tuples.
	Each tuple contains four entries:
	- the token
	- the part of speech
	- the phrase chunking tag
	- the named entity tag

	For example, the first two entries in the
	returned result for 'train.txt' are:

	> train_data = read_data('train.txt')
	> train_data[:2]
	[[('EU', 'NNP', 'I-NP', 'I-ORG'),
	  ('rejects', 'VBZ', 'I-VP', 'O'),
	  ('German', 'JJ', 'I-NP', 'I-MISC'),
	  ('call', 'NN', 'I-NP', 'O'),
	  ('to', 'TO', 'I-VP', 'O'),
	  ('boycott', 'VB', 'I-VP', 'O'),
	  ('British', 'JJ', 'I-NP', 'I-MISC'),
	  ('lamb', 'NN', 'I-NP', 'O'),
	  ('.', '.', 'O', 'O')],
	 [('Peter', 'NNP', 'I-NP', 'I-PER'), ('Blackburn', 'NNP', 'I-NP', 'I-PER')]]
	"""
	all_sent = []
	with open(filename, 'r') as fp:
		sent = []
		for line in fp:
			if line == "-DOCSTART- -X- -X- O\n":
				pass
			elif line == '\n':
				if len(sent) > 0:
					all_sent.append(sent)
				sent = []
			else:
				t = tuple(line.split())
				sent.append(t)
		all_sent.append(sent)

	return all_sent

def make_feature_dicts(data,
					   token=True,
					   caps=True,
					   pos=True,
					   chunk=True,
					   context=True):
	"""
	Create feature dictionaries, one per token. Each entry in the dict consists of a key (a string)
	and a value of 1.
	Also returns a numpy array of NER tags (strings), one per token.

	See a3_test.

	The parameter flags determine which features to compute.
	Params:
	data.......the data returned by read_data
	token......If True, create a feature with key 'tok=X', where X is the *lower case* string for this token.
	caps.......If True, create a feature 'is_caps' that is 1 if this token begins with a capital letter.
			   If the token does not begin with a capital letter, do not add the feature.
	pos........If True, add a feature 'pos=X', where X is the part of speech tag for this token.
	chunk......If True, add a feature 'chunk=X', where X is the chunk tag for this token
	context....If True, add features that combine all the features for the previous and subsequent token.
			   E.g., if the prior token has features 'is_caps' and 'tok=a', then the features for the
			   current token will be augmented with 'prev_is_caps' and 'prev_tok=a'.
			   Similarly, if the subsequent token has features 'is_caps', then the features for the
			   current token will also include 'next_is_caps'.
	Returns:
	- A list of dicts, one per token, containing the features for that token.
	- A numpy array, one per token, containing the NER tag for that token.
	"""
	ld = []
	fd = []
	ner = []
	for token_list in data:
		id = []
		for t in token_list:
			d_t = {}
			if token == True:
				lower = t[0].lower()
				key = 'tok=' + lower
				d_t[key] = 1
			if caps == True:
				if (t[0][0]).isupper() == True:
					key = 'is_caps'
					d_t[key] = 1
			if pos == True:
				key = 'pos=' + t[1]
				d_t[key] = 1
			if chunk == True:
				key = 'chunk=' + t[2]
				d_t[key] = 1
			ner.append(t[3])
			id.append(d_t)
		ld.append(id)

	if context == True:
		for f in ld:
			for i, d in enumerate(f):
				dd = {}
				if len(f) > 1:
					if i == 0:
						dd.update(d)
						next = f[i + 1]
						nkeys = next.keys()
						for k in nkeys:
							key = 'next_' + k
							dd[key] = 1
					elif i == len(f) - 1:
						dd.update(d)
						prev = f[i - 1]
						pkeys = prev.keys()
						for k in pkeys:
							key = 'prev_' + k
							dd[key] = 1
					elif i > 0 and i < len(f) - 1:
						dd.update(d)
						prev = f[i - 1]
						next = f[i + 1]
						pkeys = prev.keys()
						nkeys = next.keys()
						for k in pkeys:
							key = 'prev_' + k
							dd[key] = 1
						for k in nkeys:
							key = 'next_' + k
							dd[key] = 1
					fd.append(dd)
				else:
					fd.extend(f)
	else:
		for l in ld:
			fd.extend(l)

	return fd,np.array(ner)


def confusion(true_labels, pred_labels):
	"""
	Create a confusion matrix, where cell (i,j)
	is the number of tokens with true label i and predicted label j.

	Params:
	  true_labels....numpy array of true NER labels, one per token
	  pred_labels....numpy array of predicted NER labels, one per token
	Returns:
	A Pandas DataFrame (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
	See Log.txt for an example.
	"""

	u_labels = np.unique(true_labels)
	p_labels = np.unique(pred_labels)
	df = pd.DataFrame(index=u_labels, columns=p_labels)

	for l in u_labels:
		t = np.argwhere(true_labels == l)
		t = np.ndarray.flatten(t)
		pred = np.take(pred_labels, t)
		pred = np.ndarray.flatten(pred)
		cd = Counter(pred)
		for col in p_labels:
			df.set_value(l, col, cd[col])
	return df

def evaluate(confusion_matrix):
	"""
	Compute precision, recall, f1 for each NER label.
	The table should be sorted in ascending order of label name.
	If the denominator needed for any computation is 0,
	use 0 as the result.  (E.g., replace NaNs with 0s).

	NOTE: you should implement this on your own, not using
		  any external libraries (other than Pandas for creating
		  the output.)
	Params:
	  confusion_matrix...output of confusion function above.
	Returns:
	  A Pandas DataFrame. See Log.txt for an example.
	"""
	pd.set_option('precision', 6)
	cols = confusion_matrix.columns
	crows = confusion_matrix.index
	confusion_matrix = confusion_matrix.fillna(value=0)
	rows = ['precision', 'recall', 'f1']
	df = pd.DataFrame(index=rows, columns=cols)

	for c in cols:
		tp = confusion_matrix.get_value(c, c)
		fp = 0
		for i in crows:
			if i != c:
				fp += confusion_matrix.get_value(i, c)
			d1 = tp + fp

			if d1 == 0 or np.isnan(d1):
				precision = 0
			else:
				precision = tp / d1
			df.set_value(rows[0], c, precision)

			fn = 0
			for cl in cols:
				if cl != c:
					fn += confusion_matrix.get_value(c, cl)
			d2 = tp + fn

			if d2 == 0 or np.isnan(d2):
				recall = 0
			else:
				recall = tp / d2
			df.set_value(rows[1], c, recall)

			d3 = precision + recall
			if d3 == 0 or np.isnan(d3):
				f1 = 0
			else:
				f1 = 2 * precision * recall / d3
			df.set_value(rows[2], c, f1)

	return df

def average_f1s(evaluation_matrix):
	"""
	Returns:
	The average F1 score for all NER tags, 
	EXCLUDING the O tag.
	"""
	s = 0
	for c in evaluation_matrix.columns:
		if c != 'O':
			s += evaluation_matrix.get_value('f1', c)
	if (len(evaluation_matrix.columns)) == 1:
		avg = s
	else:
		avg = s / (len(evaluation_matrix.columns)-1)
	return avg

def evaluate_combinations(train_data, test_data):
	"""
	Run 16 different settings of the classifier, 
	corresponding to the 16 different assignments to the
	parameters to make_feature_dicts:
	caps, pos, chunk, context
	That is, for one setting, we'll use
	token=True, caps=False, pos=False, chunk=False, context=False
	and for the next setting we'll use
	token=True, caps=False, pos=False, chunk=False, context=True

	For each setting, create the feature vectors for the training
	and testing set, fit a LogisticRegression classifier, and compute
	the average f1 (using the above functions).

	Returns:
	A Pandas DataFrame containing the F1 score for each setting, 
	along with the total number of parameters in the resulting
	classifier. This should be sorted in descending order of F1.
	(See Log.txt).

	Note1: You may find itertools.product helpful for iterating over
	combinations.

	Note2: You may find it helpful to read the main method to see
	how to run the full analysis pipeline.
	"""
	caps = [False,True]
	pos = [False,True]
	chunk = [False,True]
	context = [False,True]

	argnames = ['caps', 'pos', 'chunk', 'context']
	cols = ['f1', 'n_params'] + argnames
	df = pd.DataFrame(columns=cols)
	combs = product(caps, pos, chunk, context)
	for i, c in enumerate(combs):
		dicts, labels = make_feature_dicts(train_data, True, *c)
		test_dicts, test_labels = make_feature_dicts(test_data, True, *c)
		vec = DictVectorizer()
		X = vec.fit_transform(dicts)
		clf = LogisticRegression()
		clf.fit(X, labels)
		X_test = vec.transform(test_dicts)
		preds = clf.predict(X_test)
		cf = (len(clf.coef_))
		dm = X_test.shape[1]
		n_params = cf * dm
		confusion_matrix = confusion(test_labels, preds)
		evaluation_matrix = evaluate(confusion_matrix)
		avgf1 = average_f1s(evaluation_matrix)
		val = [avgf1, n_params] + [c1 for c1 in c]
		df.loc[i] = val
	df['n_params'] = df['n_params'].astype(int)
	df = df.sort_values(by='f1', ascending=False)

	return df

if __name__ == '__main__':
	"""
	This method is done for you.
	See Log.txt for expected output.
	"""
	download_data()
	train_data = read_data('train.txt')
	dicts, labels = make_feature_dicts(train_data,
								   token=True,
								   caps=True,
								   pos=True,
								   chunk=True,
								   context=True)
	vec = DictVectorizer()
	X = vec.fit_transform(dicts)
	print('training data shape: %s\n' % str(X.shape))
	clf = LogisticRegression()
	clf.fit(X, labels)

	test_data = read_data('test.txt')
	test_dicts, test_labels = make_feature_dicts(test_data,
												 token=True,
												 caps=True,
												 pos=True,
												 chunk=True,
												 context=True)
	X_test = vec.transform(test_dicts)
	print('testing data shape: %s\n' % str(X_test.shape))

	preds = clf.predict(X_test)

	confusion_matrix = confusion(test_labels, preds)
	print('confusion matrix:\n%s\n' % str(confusion_matrix))

	evaluation_matrix = evaluate(confusion_matrix)
	print('evaluation matrix:\n%s\n' % str(evaluation_matrix))

	print('average f1s: %f\n' % average_f1s(evaluation_matrix))

	combo_results = evaluate_combinations(train_data, test_data)
	print('combination results:\n%s' % str(combo_results))


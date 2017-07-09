import numpy as np
import pandas as pd
import urllib.request
import tensorflow as tf
from nltk.corpus import brown
from tensorflow.contrib import rnn
from gensim.models import word2vec


def download_data():
	url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
	urllib.request.urlretrieve(url, 'train.txt')
	url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'
	urllib.request.urlretrieve(url, 'test.txt')

def read_data(filename):
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

def generate_data(data, tag_to_num):
	X = []
	Y = []
	brown_sents = brown.sents()
	w2v_model = word2vec.Word2Vec(brown_sents, size=300, window=5, min_count=5)
	for l in data:
		for t,tup in enumerate(l):
			y = np.zeros((5, 1))
			if tup[0] in w2v_model.wv.vocab.keys():
				w2v_arr = w2v_model.wv[tup[0]]
				X.append(w2v_arr[:300])
				y[tag_to_num[tup[3]]] = 1
				Y.append(y)
	X = np.asarray(X)
	X = np.reshape(X,(X.shape[0],X.shape[1]))
	Y = np.asarray(Y)
	Y = np.reshape(Y,(Y.shape[0],Y.shape[1]))
	return X,Y

def RNN(x, weights, biases):
	x = tf.unstack(x, n_steps, 1)
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

def confusion(y_ids, pred_ids, num_to_tag):
	result = []
	for i in range(0, 5):
		result.append(num_to_tag[i])
	preds = []
	true_labels = []
	for i in pred_ids:
		preds.append(num_to_tag[i])
	for i in y_ids:
		true_labels.append(num_to_tag[i])
	test = sorted(result)
	result = np.zeros((len(test), len(test)))
	for true_label_idx, i in enumerate(sorted(test)):
		for pred_label_idx, j in enumerate(sorted(test)):
			for a, b in zip(true_labels, preds):
				if a == i and b == j:
					result[true_label_idx][pred_label_idx] += 1
	return pd.DataFrame(result.astype(np.int32), test, test)

def evaluate(confusion_matrix, num_to_tag):
	precision_list = []
	recall_list = []
	f1_list = []
	result = []
	for i in range(len(num_to_tag)):
		result.append(num_to_tag[i])
	for idx, item in enumerate(sorted(result)):
		precision = np.diag(confusion_matrix)[idx] / np.sum(confusion_matrix, axis=0)[item]
		recall = np.diag(confusion_matrix)[idx] / np.sum(confusion_matrix, axis=1)[item]
		f1 = 0
		if precision + recall > 0:
			f1 = 2 * (precision * recall) / (precision + recall)
		f1_list.append(f1)
		recall_list.append(recall)
		precision_list.append(precision)
	return pd.DataFrame(np.array([precision_list, recall_list, f1_list]), index=['precision', 'recall', 'f1'],
						columns=sorted(result))


def average_f1s(evaluation_matrix, num_to_tag):
	f1 = 0.0
	count = 0
	result = []
	for i in range(len(num_to_tag)):
		result.append(num_to_tag[i])
	for idx, item in enumerate(sorted(result)):
		if item.lower() != 'o':
			f1 += evaluation_matrix.iloc[2][idx]
			count += 1
	return f1 / count



if __name__ == '__main__':
	# Parameters
	learning_rate = 10
	training_iters = 1000
	batch_size = 128
	display_step = 10

	# Network Parameters
	n_input = 15
	n_steps = 20
	n_hidden = 20
	n_classes = 5

	x = tf.placeholder("float32", [None, n_steps, n_input])
	y = tf.placeholder("float32", [None, n_classes])

	weights = {
		'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
	}
	biases = {
		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	predicted = RNN(x, weights, biases)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	init = tf.global_variables_initializer()


	tagnames = ['I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
	num_to_tag = dict(enumerate(tagnames))
	tag_to_num = {v: k for k, v in num_to_tag.items()}
	download_data()
	X_train, Y_train = generate_data(read_data('train.txt'), tag_to_num)
	print('Training data shape: %s\n' % str(X_train.shape))
	X_test, Y_test = generate_data(read_data('test.txt'), tag_to_num)
	print('Testing data shape: %s\n' % str(X_test.shape))
	with tf.Session() as sess:
		sess.run(init)
		step = 0
		while step * batch_size < training_iters:
			batch_x = X_train[step * batch_size: step * batch_size + batch_size]
			batch_y = Y_train[step * batch_size: step * batch_size + batch_size]
			batch_x = batch_x.reshape((batch_size, n_steps, n_input))
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
			if step % display_step == 0:
				acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
				loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
			step += 1
		test_data = X_test.reshape((X_test.shape[0], n_steps, n_input))
		test_label = Y_test
		y_p = tf.argmax(predicted, 1)

		val_acc, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_data, y: test_label})
		y_true = np.argmax(test_label, 1)
		confusion_matrix = confusion(y_pred, y_true, num_to_tag)
		print('confusion matrix:\n%s\n' % str(confusion_matrix))
		evaluation_matrix = evaluate(confusion_matrix, num_to_tag)
		print('evaluation matrix:\n%s\n' % str(evaluation_matrix))
		print('average f1s: %f\n' % average_f1s(evaluation_matrix, num_to_tag))

		with open('output2.txt','w') as fp:
			fp.write('training data shape: %s\n' % str(X_train.shape))
			fp.write('Testing data shape: %s\n' % str(X_test.shape))
			fp.write('confusion matrix:\n%s\n' % str(confusion_matrix))
			fp.write('evaluation matrix:\n%s\n' % str(evaluation_matrix))
			fp.write('average f1s: %f\n' % average_f1s(evaluation_matrix, num_to_tag))
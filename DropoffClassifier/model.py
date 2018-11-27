from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
path = 'data/'

n_train_in = path + 'train_in.npy'
n_train_out = path + 'train_out.npy'
n_test_in = path + 'test_in.npy'
n_test_out = path + 'test_out.npy'

train_in = np.load(n_train_in)
train_out = np.load(n_train_out)
test_in = np.load(n_test_in)
test_out = np.load(n_test_out)


def svm_model(model_name, load=True):
	if load:
		print("LOADING SVM")
		clf = pickle.load(open(model_name, 'rb'))
	else:	
		clf = SVC()
		clf.fit(train_in, train_out)
		pickle.dump(clf, open(model_name, 'wb'))
	print("TESTING")
	print(clf.score(test_in, test_out))

def mlp_model(model_name, load=True):
	if load:
		print("LOADING MLP")
		clf = pickle.load(open(model_name, 'rb'))
	else:
		print("INITIALIZING MLP CLASSIFIER")
		clf = MLPClassifier(hidden_layer_sizes = (64, 128, 64), learning_rate='invscaling')
		clf.fit(train_in, train_out)
		pickle.dump(clf, open(model_name, 'wb'))
	print("TESTING")
	print(clf.score(test_in, test_out))

def tree_model(model_name, load=True):
	if load:
		print("LOADING TREE")
		clf = pickle.load(open(model_name, 'rb'))
	else:
		print("TRAINING TREE")
		clf = DecisionTreeClassifier(max_depth = 8 , min_samples_leaf=2)
		clf.fit(train_in, train_out)
	print("TESTING")
	s = clf.score(test_in, test_out)
	print(s)
	if s > 0.75:
		pickle.dump(clf, open(model_name, 'wb'))

def SGD_model(model_name, load=True):
	if load:
		print("LOADING SGD")
		clf = pickle.load(open(model_name, 'rb'))
	else:
		print("INITIALIZING SGD CLASSIFIER")
		n = train_in.size
		clf = SGDClassifier(penalty='l2', learning_rate='constant',eta0 =0.001)
		clf.fit(train_in, train_out)
	print("TESTING")
	s = clf.score(test_in, test_out)
	print(s)
	if s > 0.75:
		pickle.dump(clf, open(model_name, 'wb'))


#svm_model("clf.sav", True) # 0.72
#mlp_model("mlp2.sav", True) # 0.7842
#mlp_model("mlp.sav", True) # 0.7899
#tree_model('tree.sav', False) # 0.722
#SGD_model('sgd.sav', False) # 0.70
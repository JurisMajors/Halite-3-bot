from sklearn.svm import SVC
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
		clf = pickle.load(open(model_name, 'rb'))
	else:	
		clf = SVC()
		clf.fit(train_in, train_out)
		pickle.dump(clf, open(model_name, 'wb'))

	print(clf.score(test_in, test_out))

def mlp_model(model_name, load=True):
	pass

svm_model("clf.sav")

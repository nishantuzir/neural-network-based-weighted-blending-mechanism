import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import svm
from xgboost import XGBClassifier
from xgboost import plot_importance 
#from matplotlib import pyplot

def blender(t,s,e):
	traindata = pd.read_csv(t, header=None)
	#testdata = pd.read_csv('/Users/nishantuzir/Downloads/ponzi_chat_traffic_classification/ponzi_traffic_test.csv'

	X = traindata.iloc[:,s:e]
	Y = traindata.iloc[:,e]
	#C = testdata.iloc[:,38]
	#T = testdata.iloc[:,0:38]

	scaler = Normalizer().fit(X)
	trainX = scaler.transform(X)
	#scaler = Normalizer().fit(T)
	#testT = scaler.transform(T)
	traindata = np.array(trainX)
	trainlabel = np.array(Y)
	#testdata = np.array(testT)
	#testlabel = np.array(C)
	seed = 1337
	test_size = 0.30
	x_train,x_test,y_train,y_test = train_test_split(traindata,trainlabel,test_size = test_size, random_state = seed)

	modellr = LogisticRegression()
	modelnb = GaussianNB()
	modeldt = DecisionTreeClassifier()
	modelrf = RandomForestClassifier()

	models = [modellr,modelnb,modeldt,modelrf]
	predictions = pd.DataFrame(columns=[i for i in range(len(models))])
	for j,model in enumerate(models):
		predictions[j] = model.fit(x_train, y_train).predict_proba(x_test)[:,1]

	predictions = np.array(predictions)
	from sklearn.neural_network import MLPClassifier
	modelnn = MLPClassifier(hidden_layer_sizes=(100,),max_iter=500)
	modelnn.fit(predictions,y_test)

	expected = y_test
	predicted = modelnn.predict(predictions)
	global accuracy,recall,precision,f1,cm
	accuracy = accuracy_score(expected, predicted)
	recall = recall_score(expected, predicted, average="binary")
	precision = precision_score(expected, predicted , average="binary")
	f1 = f1_score(expected, predicted , average="binary")
	cm = metrics.confusion_matrix(expected, predicted)
	print(cm)
	#pyplot.matshow(cm)
	#pyplot.title('Confusion matrix')
	#pyplot.colorbar()
	#pyplot.ylabel('True label')
	#pyplot.xlabel('Predicted label')
	#pyplot.show()
	print("Accuracy")
	print("%.3f" %accuracy)
	print("precision")
	print("%.3f" %precision)
	print("recall")
	print("%.3f" %recall)
	print("f-score")
	print("%.3f" %f1)

	return modelnn

if __name__ == "__main__":
	import sys
	import argparse
	parser = argparse.ArgumentParser(description='neural network based wighted blender')
	parser.add_argument('-t', '--train', default=None, help='specify path of train data')
	parser.add_argument('-e', '--end', type=int,default=10, help='specify the end column of the train dataset; this value will be excluded because python is upper limit exclusive')
	parser.add_argument('-s', '--start',type=int ,default=0, help='specify the start column of the train dataset; this value will not be excluded because python is not lower limit exclusive')
	args = parser.parse_args()
	if args.train:
		blender(args.train,args.start,args.end)
	else:
		parser.print_help()






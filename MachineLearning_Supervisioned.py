import rdflib
import string
import pandas
import random
#from pandas.plotting import scatter_matrix
#from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
#url = 'breast-cancer-wisconsin.txt'
#names = ['id', 'clump_thickness', 'u_cell_size', 'u_cell_shape', 'marg_adhesion', 'epi_cell_size', 'bare_nuclei', 'bland_chrom', 'norm_nucl', 'mitoses', 'class']
#names =['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10']
#dataset = pandas.read_csv(url, header=None, names=names, prefix='Feature', index_col=0) #prefix='Feature'

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names) #prefix='Feature'




#Statistic information dataset
print(dataset.shape)
print(dataset.describe())
print(dataset.groupby('class').size())
plt.show()



# Plot the statistic of dataset
#UNIVARIABLE
dataset.hist()
plt.show()

#*******MULTIVARIABLE*********
#scatter_matrix(dataset)
plt.show()


array = dataset.values
#Select the features that will be used in the model
X = array[:,0:4] 
Y = array[:,4]
validation_size = 0.40
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)
predictions = CART.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
#from sknn.mlp import Classifier, Layer

degree = [2, 4]

soundData = pd.read_csv('MFCC_Y&N_features.csv')
soundData2 = pd.read_csv('MFCC_reclean_features.csv')

X = soundData.loc[0:(soundData.shape[0] - 4), '0': (len(soundData.columns) - 2).__str__()]
X2 = soundData2.loc[0:(soundData2.shape[0] - 5), '0': (len(soundData2.columns) - 2).__str__()]
ytarget = soundData['Target']
ytarget2 = soundData2['Target']

X_train, X_test, y_train, y_test = train_test_split(X, ytarget, test_size=0.5, shuffle=True, random_state=8)

pca = PCA(n_components=16, svd_solver='full')
knn = KNeighborsClassifier(n_neighbors=5)

svm_pipe = make_pipeline(StandardScaler(), pca, SVC(random_state=1, cache_size=1000))

svm_param_grid =[{'svc__C':[0.1,1,10],
                  'svc__kernel':['poly', 'rbf', 'linear', 'sigmoid'],
                  'svc__degree': degree,
                  'pca__n_components': [2,1,20]}]

gs_SVM = GridSearchCV(estimator=svm_pipe,
                      param_grid=svm_param_grid,
                      scoring='accuracy',
                      refit=True,
                      cv=3, n_jobs=1, verbose=0)
scores_SVM = cross_val_score(gs_SVM, X_train, y_train, scoring='accuracy', cv=10, verbose=0)
SVM_model = gs_SVM.fit(X_train, y_train)
print("SVM PREDICT ON MY OWN DATA")
print(gs_SVM.predict(X2))
print(gs_SVM.score(gs_SVM.predict(X2), ytarget))
print("Mean Accuracy for SVM: {:f}".format(np.mean(scores_SVM)))
print("Stdev of Accuracy for SVM: {:f}".format(np.std(scores_SVM)))
print("Best score from SVM gridsearch: %s" % SVM_model.best_score_)





lr = LogisticRegression(max_iter=1000,solver='newton-cg')
lr_pipe = make_pipeline(StandardScaler(), pca, lr)
lr_param_grid =[{'logisticregression__C':[1e-2, 1e-1, 1],
                 'pca__n_components': [2,1,20]}]

gs_Lin = GridSearchCV(estimator=lr_pipe,
                  param_grid=lr_param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=3, verbose=0)
scores_Lr = cross_val_score(gs_Lin, X_train, y_train, scoring='accuracy', cv=10, verbose=0)
#print(gs_Lin.predict(X2))

print("Mean Accuracy for LR: {:f}".format(np.mean(scores_Lr)))
print("Stdev of Accuracy for LR: {:f}".format(np.std(scores_Lr)))

KNNclassifier_pipline = make_pipeline(StandardScaler(), pca, knn)

KNN_param_grid = [{'kneighborsclassifier__n_neighbors': [1,1,30],
                   'pca__n_components': [2,1,20]}]

gs_KNN = GridSearchCV(estimator=KNNclassifier_pipline,
                      param_grid=KNN_param_grid,
                      scoring='accuracy',
                      refit=True,
                      cv=3, verbose=0)

scores_KNN = cross_val_score(estimator=gs_KNN, X=X_train, y=y_train, cv=10, n_jobs=1, verbose=1)
modelKNN = gs_KNN.fit(X_train, y_train)
predictionsKNN = modelKNN.score(X_test, y_test)
print(gs_KNN.predict(X2))

print("Mean KNN CV scores: %s" % scores_KNN.mean())
print("KNN model score: %s" % modelKNN.best_score_)


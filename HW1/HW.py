import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import scatterplot
from sklearn import neighbors
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metr
from sklearn import preprocessing
import math
path1= 'C:\\Users\\HP\\Desktop\\EE559\\HW1\\vertebral_column_data'
path=path1+'/column_2C.dat'                                                       # create dirtory
data = pd.read_table(path,header=None,sep='\s+')
data.columns = ['PI','PT','LLA','SS','PR','GOS','class']                        # change the columns'name of DATAFRAME
#
# # sns.pairplot(data, hue="class",diag_kind="kde")
# plt.show()

train_AB = data.loc[data['class']=='AB'].head(140)
train_NO = data.loc[data['class']=='NO'].head(70)
train = train_AB.append(train_NO)

test_AB = data.loc[data['class']=='AB'][140:]
test_NO = data.loc[data['class']=='NO'][70:]
test = test_AB.append(test_NO)


def knn( X_train,X):
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.multiply(np.dot(X, X_train.T), -2)
    print (dists.shape)
    sq1 = np.sum(np.square(X), axis=1, keepdims=True)
    sq2 = np.sum(np.square(X_train), axis=1)
    dists = np.add(dists, sq1)
    dists = np.add(dists, sq2)
    dists = dists.astype('float32')
    dists = np.sqrt(dists)
    return dists

def predict_labels(dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = np.array(y_train)[np.argsort(dists[i])[:k]]
        y_pred[i] = np.argmax(np.bincount(closest_y))
    return y_pred

train1 = train.values
print (type(train1[1][6]))
train_1 = np.delete(train1,6,axis =1)
y_train = []
y_test  = []

for i in range(210):
    if train1[i][6] == 'AB':
        y_train.append(1)
    elif train1[i][6] == 'NO':
        y_train.append(0)
    else:
        break
print (len(y_train))


test1 = test.values
test_1 = np.delete(test1,6,axis =1)
for i in range(100):
    if test1[i][6] == 'AB':
        y_test.append(1)
    elif test1[i][6] == 'NO':
        y_test.append(0)
    else:
        break
print (len(y_test))

dists = knn(train_1 ,test_1 )
print (dists)


k_ = range(1,211,3)
final_result = []
for i in range(len(k_)):
    y_pred = predict_labels(dists, k=k_[i])
    for j in range(len(y_pred)):
        num_correct = np.sum(y_pred == y_test[j])
    accuracy = float(num_correct) / len(y_pred)
    print('Got %d / %d correct => accuracy: %f, when k is %d' % (num_correct, len(y_pred), accuracy,k_[i]))
    final_result.append(accuracy)
plt.errorbar(k_, final_result)
plt.title('test of K')
plt.xlabel('K')
plt.ylabel('Accuracy')
# plt.show()

best_k = k_[final_result.index(max(final_result ))]
print ('The best k is %d' %(best_k ))
y_pred_best = predict_labels(dists,k = best_k)

C2 = confusion_matrix(y_test, y_pred_best, labels=None, sample_weight=None)
print ('The confusion matrix is')
print (C2)
TP = C2[0][0]
TN=  C2[1][1]
FP=  C2[1][0]
FN=  C2[0][1]
TPR = TP /(TP + FN)
TNR = TN /(TN + FP)
precision = TP/(TP+FP)
recall=TP/(TP+FN)
f1_score = 2*precision * recall/(precision + recall )
print ('The Ture positive rate is %f'%(TPR))
print ('The Ture negative rate is %f'%(TNR))
print ('precision is %f'%(precision))
print ('The f1 score is %f'%(f1_score))


sns.heatmap(C2,annot=True)
# plt.show()


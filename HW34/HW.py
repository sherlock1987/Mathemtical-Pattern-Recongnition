import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
import seaborn as sns
from seaborn import scatterplot
from sklearn import neighbors
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metr
from sklearn import preprocessing
import math
import os
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from numpy import linalg as la
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve,auc
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.preprocessing import label_binarize



path= 'C:\\Users\\HP\\Desktop\\EE559\\HW34\\AReM'
def create_samples(path):
    filename_whole = []
    for filename in os.listdir(path):
        if ('.pdf' in filename)==False:
          filename_whole.append(filename)
    filename_whole.sort
    test = []
    train = []
    data = []
    for i in range(len(filename_whole)):
        path_1 = path +'\\'+filename_whole[i]
        if i <=1 :
            print (path_1)
            filename1 = []
            for filename in os.listdir(path_1):
                filename1.append(filename)
            filename1.sort()
            for j in range(0,2,1):
                path1_1 = path_1 + '\\' + filename1[j]
                a = pd.read_csv(path1_1, sep=None, engine='python', skiprows=range(0, 4, 1))
                b = a.values
                test.append(b)
                data.append(b)
                print (path1_1)
            for j in range(2,len(filename1),1):
                if ((i==1)&(j==3))==True:
                    path1_1 = path_1 + '\\' + filename1[j]
                    a = pd.read_csv(path1_1, sep=' ', engine='python', skiprows=range(0, 5, 1))
                    a = a.drop('Unnamed: 7', axis=1)
                    b = a.values
                    train.append(b)
                    data.append(b)
                    print (path1_1)
                else:
                    path1_1 = path_1 + '\\' + filename1[j]
                    a = pd.read_csv(path1_1, sep=None, engine='python', skiprows=range(0, 4, 1))
                    b = a.values
                    train.append(b)
                    data.append(b)
                    print(path1_1)
        else:
            print(path_1)
            filename1 = []
            for filename in os.listdir(path_1):
                filename1.append(filename)
            filename1.sort()
            for j in range(0, 3, 1):
                path1_1 = path_1 + '\\' + 'dataset'+str(j+1)+'.csv'
                print (path1_1)

                a = pd.read_csv(path1_1, sep=None, engine='python', skiprows=range(0, 4, 1))
                b = a.values
                test.append(b)
                data.append(b)

            for j in range(3, len(filename1), 1):
                path1_1 = path_1 + '\\' + 'dataset'+str(j+1)+'.csv'
                a = pd.read_csv(path1_1, sep=',', engine='python', skiprows=range(0, 4, 1), error_bad_lines=False)
                b = a.values
                train.append(b)
                data.append(b)
                print (path1_1)

    print (len(test))
    print (len(train))
    print (len(data))
    return test,train,data


def create_samples_multi(path):
    filename_whole = []
    for filename in os.listdir(path):
        if ('.pdf' in filename)==False:
          filename_whole.append(filename)
    filename_whole.sort
    test = []
    train = []
    data = []
    test_classes = []
    train_classes= []
    for i in range(len(filename_whole)):
        path_1 = path +'\\'+filename_whole[i]
        if i <=1 :
            print (path_1)
            filename1 = []
            for filename in os.listdir(path_1):
                filename1.append(filename)
            filename1.sort()
            for j in range(0,2,1):
                path1_1 = path_1 + '\\' + filename1[j]
                a = pd.read_csv(path1_1, sep=None, engine='python', skiprows=range(0, 4, 1))
                b = a.values
                test.append(b)
                data.append(b)
                print (path1_1)
                test_classes.append(filename_whole[i])
            for j in range(2,len(filename1),1):
                if ((i==1)&(j==3))==True:
                    path1_1 = path_1 + '\\' + filename1[j]
                    a = pd.read_csv(path1_1, sep=' ', engine='python', skiprows=range(0, 5, 1))
                    a = a.drop('Unnamed: 7', axis=1)
                    b = a.values
                    train.append(b)
                    data.append(b)
                    train_classes.append(filename_whole[i])
                    print (path1_1)
                else:
                    path1_1 = path_1 + '\\' + filename1[j]
                    a = pd.read_csv(path1_1, sep=None, engine='python', skiprows=range(0, 4, 1))
                    b = a.values
                    train.append(b)
                    data.append(b)
                    train_classes.append(filename_whole[i])
                    print(path1_1)
        else:
            print(path_1)
            filename1 = []
            for filename in os.listdir(path_1):
                filename1.append(filename)
            filename1.sort()
            for j in range(0, 3, 1):
                path1_1 = path_1 + '\\' + 'dataset'+str(j+1)+'.csv'
                print (path1_1)

                a = pd.read_csv(path1_1, sep=None, engine='python', skiprows=range(0, 4, 1))
                b = a.values
                test.append(b)
                data.append(b)
                test_classes.append(filename_whole[i])

            for j in range(3, len(filename1), 1):
                path1_1 = path_1 + '\\' + 'dataset'+str(j+1)+'.csv'
                a = pd.read_csv(path1_1, sep=',', engine='python', skiprows=range(0, 4, 1), error_bad_lines=False)
                b = a.values
                train.append(b)
                data.append(b)
                train_classes.append(filename_whole[i])
                print (path1_1)

    print (len(test))
    print (len(train))
    print (len(data))
    return test,train,data,train_classes ,test_classes

def c_1():
    print('There are a lot of features used in time series')
    print('like correlation structure, distribution，entropy，stationarity，scaling properties')

def feature_extraction(data):
     aa = []
     for i in range(6):
         for j in range(8):
             if j == 1:
                 aa.append('min_' + str(i + 1))
             elif j == 2:
                 aa.append('max_' + str(i + 1))
             elif j == 3:
                 aa.append('mean_' + str(i + 1))
             elif j == 4:
                 aa.append('median_' + str(i + 1))
             elif j == 5:
                 aa.append('standard deviation_' + str(i + 1))
             elif j == 6:
                 aa.append('1st quart_' + str(i + 1))
             elif j == 7:
                 aa.append('3rd quart_' + str(i + 1))
             else:
                 pass

     result = pd.DataFrame(columns=aa)
     for k in range(88):
         bb = []
         for i in range(6):
             for j in range(7):
                 if j == 0:
                     bb.append(np.min(data[k][:,i+1]))
                 elif j == 1:
                     bb.append(np.max(data[k][:,i+1]))
                 elif j == 2:
                     bb.append(np.mean(data[k][:,i+1]))
                 elif j == 3:
                     bb.append(np.median(data[k][:,i+1]))
                 elif j == 4:
                     bb.append(np.std(data[k][:,i+1]))
                 elif j == 5:
                     bb.append(np.percentile(data[k][:,i+1],25))
                 elif j == 6:
                     bb.append(np.percentile(data[k][:,i+1],75))
                 else:
                     pass
         result.loc[k] = bb
     print (result)
     # print (np.max(data[0][:,1]))
     # print (np.min(data[0][:,1]))
     # print (np.mean(data[0][:,1]))
     # print (np.median(data[0][:,1]))
     # print (np.std(data[0][:,1]))
     # print (np.percentile(data[0][:,1],25))
     # print (np.percentile(data[0][:,1],75))

def d_1(train):
    data_list = []
    column_index = []
    for i in [1,2,6]:
        mean_1=[]
        std_1 = []
        percentile_1st_1 = []
        class_1 = []
        for j in range(9):
            mean = np.mean(train[j][:,i])
            std = np.std(train[j][:,i])
            percentile_1st = np.percentile(train[j][:,i],25)
            mean_1.append(mean)
            std_1.append(std)
            percentile_1st_1.append(percentile_1st)
            class_1.append(0)
        for j in range(9,69,1):
            mean = np.mean(train[j][:,i])
            std = np.std(train[j][:,i])
            percentile_1st = np.percentile(train[j][:,i],25)
            mean_1.append(mean)
            std_1.append(std)
            percentile_1st_1.append(percentile_1st)
            class_1.append(1)
        data_list.append(mean_1)
        data_list.append(std_1)
        data_list.append(percentile_1st_1)
        column_index.append(str(i)+'_mean')
        column_index.append(str(i)+'_std')
        column_index.append(str(i)+'_percentile_1st')
    data_list.append(class_1)
    column_index.append('class')
    data_result = np.array(data_list)
    print (data_result.shape)
    data_df = DataFrame (data_result.T,columns=column_index )
    sns.pairplot(data_df,hue = 'class',vars=['1_mean', '1_std', '1_percentile_1st', '2_mean', '2_std', '2_percentile_1st', '6_mean', '6_std', '6_percentile_1st'])

def d_2(train,num):
    piece = round(480/num)
    for i in [1,2,6]:
        mean_1=[]
        std_1 = []
        percentile_1st_1 = []
        class_1 = []
        data_list = []
        for j in range(9):
            for k in range (num):
                if k == (num-1):
                    num_1 = len(train[j][:,1])
                    mean = np.mean(train[j][k*piece:num_1,i])
                    std = np.std(train[j][k*piece:num_1,i])
                    percentile_1st = np.percentile(train[j][k*piece:num_1,i],25)
                    mean_1.append(mean)
                    std_1.append(std)
                    percentile_1st_1.append(percentile_1st)
                    class_1.append('bent')
                else:
                    mean = np.mean(train[j][k * piece:(k + 1) * piece, i])
                    std = np.std(train[j][k * piece:(k + 1) * piece, i])
                    percentile_1st = np.percentile(train[j][k * piece:(k + 1) * piece, i], 25)
                    mean_1.append(mean)
                    std_1.append(std)
                    percentile_1st_1.append(percentile_1st)
                    class_1.append('bent')
        for j in range(9,69,1):
            for k in range (num):
                if k == (num - 1):
                    num_1 = len(train[j][:, 1])
                    mean = np.mean(train[j][k * piece:num_1, i])
                    std = np.std(train[j][k * piece:num_1, i])
                    percentile_1st = np.percentile(train[j][k * piece:num_1, i], 25)
                    mean_1.append(mean)
                    std_1.append(std)
                    percentile_1st_1.append(percentile_1st)
                    class_1.append('others')
                else:
                    mean = np.mean(train[j][k * piece:(k + 1) * piece, i])
                    std = np.std(train[j][k * piece:(k + 1) * piece, i])
                    percentile_1st = np.percentile(train[j][k * piece:(k + 1) * piece, i], 25)
                    mean_1.append(mean)
                    std_1.append(std)
                    percentile_1st_1.append(percentile_1st)
                    class_1.append('others')
        print (len(mean_1))
        print (class_1)
        data_list.append(mean_1)
        data_list.append(std_1)
        data_list.append(percentile_1st_1)
        data_list.append(class_1)
        data_result = np.array(data_list)
        data_df = DataFrame (data_result.T,columns=['mean_'+str(i),'std_'+str(i),'percentile_'+str(i),'class'])
        # sns.pairplot(data_df,hue = 'class',vars=['mean_'+str(i), 'std_'+str(i),'percentile_'+str(i)])

def d_2_1(train):
    piece = 240
    data_list = []
    column_index = []
    result = {}
    for i in [1,6]:
        for k in range (2):
            mean_1 = []
            std_1 = []
            percentile_1st_1 = []
            class_1 = []

            for j in range(0,69,1):
                if j>= 9:
                    flag =1
                else:
                    flag = 0
                if (((i==6)&(k==1))==True ):
                    mean = np.mean(train[j][k * piece:(k + 1) * piece, i])
                    std = np.std(train[j][k * piece:(k + 1) * piece, i])
                    percentile_1st = np.percentile(train[j][k * piece:(k + 1) * piece, i], 25)
                    mean_1.append(mean)
                    std_1.append(std)
                    percentile_1st_1.append(percentile_1st)
                    class_1.append(flag)
                elif i==1:
                    mean = np.mean(train[j][k * piece:(k + 1) * piece, i])
                    std = np.std(train[j][k * piece:(k + 1) * piece, i])
                    percentile_1st = np.percentile(train[j][k * piece:(k + 1) * piece, i], 25)
                    mean_1.append(mean)
                    std_1.append(std)
                    percentile_1st_1.append(percentile_1st)
                    class_1.append(flag)
                else:
                    pass
            if (((i==6)&(k==1))==True )|(i==1):
                data_list.append(mean_1)
                data_list.append(std_1 )
                data_list.append(percentile_1st_1 )
            else:
                pass

    data_list.append(class_1)
    for i in [1,2,12]:
        column_index.append(str(i)+'_mean')
        column_index.append(str(i)+'_std')
        column_index.append(str(i)+'_percentile_1st')
    column_index.append('class')
    for i in range(10):
        result[column_index[i]] = data_list[i]
    data_df = DataFrame (result)
    sns.pairplot(data_df,hue = 'class',vars=['1_mean', '1_std', '1_percentile_1st', '2_mean', '2_std', '2_percentile_1st', '12_mean', '12_std', '12_percentile_1st'])

def d_3_1(train,num,sep):
    piece = round(480 / num)
    results = {}
    if sep == 0:
        sep_1= 9
    else:
        sep_1 =4
    for i in [1,2,3,4,5,6]:
        for k in range(num):
            mean_1 = []
            std_1 = []
            percentile_1st_1 = []
            class_1 = []
            for j in range(len(train)):# 69 or...
                if j>= sep_1:
                    flag =0
                else:
                    flag = 1
                if k == (num - 1):
                    num_1 = len(train[j][:, 1])
                    mean = np.mean(train[j][k * piece:num_1, i])
                    std = np.std(train[j][k * piece:num_1, i])
                    percentile_1st = np.max(train[j][k * piece:num_1, i])

                    # percentile_1st = np.percentile(train[j][k * piece:num_1, i], 75)
                    mean_1.append(mean)
                    std_1.append(std)
                    percentile_1st_1.append(percentile_1st)
                    class_1.append(flag)
                else:
                    mean = np.mean(train[j][k * piece:(k + 1) * piece, i])
                    std = np.std(train[j][k * piece:(k + 1) * piece, i])
                    percentile_1st = np.max(train[j][k * piece:(k + 1) * piece, i])

                    # percentile_1st = np.percentile(train[j][k * piece:(k + 1) * piece, i], 75)
                    mean_1.append(mean)
                    std_1.append(std)
                    percentile_1st_1.append(percentile_1st)
                    class_1.append(flag)
            results[str(i)+ '_'+str(k+1)+'_mean'] = mean_1
            results[str(i)+ '_'+str(k+1)+'_std'] = std_1
            results[str(i)+ '_'+str(k+1)+'_percentile_3st'] = percentile_1st_1
    results['class'] = class_1
    return results


def test_LogisticRegression(X_train, X_test, y_train, y_test):

    cls = LogisticRegression()
    results = cls.fit(X_train, y_train)
    scores= cls.score(X_test, y_test)
    return scores


def features_backward_selection(x_train,y_train,x_test,y_test):
    selected_features=[]
    scores_result = []
    estimator = LogisticRegression()
    for i in range(1,len(x_train[:][0])):
        scores = []
        delete_column_1 = []
        selector = RFE(estimator, n_features_to_select=i)
        selector = selector.fit(x_train, y_train)
        delete_column= (selector.support_)
        for j in range(len(delete_column)):
            if delete_column[j] == True:
                delete_column_1.append(j)
            else:
                pass
        x_train_new = np.delete(x_train,delete_column_1, axis=1)
        x_test_new = np.delete(x_test,delete_column_1, axis=1)
        sfolder = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
        for train, test in sfolder.split(x_train_new, y_train ):
            # x_train, y_train, x_test, y_test
            x_train_1 = np.delete(x_train_new,test,axis =0)
            x_test_1  = np.delete(x_train_new,train ,axis =0)
            y_train_1 = np.delete(y_train,test,axis =0)
            y_test_1  = np.delete(y_train,train ,axis =0)
            scores.append(test_LogisticRegression(x_train_1, x_test_1, y_train_1, y_test_1))
        scores_1 = np.mean(np.array(scores ))
        scores_result.append(scores_1)
    print(scores_result )
    bbb = np.argmax(np.array(scores_result))
    aaa = (np.array(scores_result))
    result_1 = len(x_train[:][0])-(bbb+1)
    result_2 = np.max(aaa)
    return result_1,result_2,x_train_new,x_test_new



def p_values(x_train,y_train):
    print(x_train.shape)
    logit_model = sm.Logit(y_train,x_train)
    result = logit_model.fit(method='ncg')
    print(result.summary2())



def find_best():
    accuracy = []
    features = []
    x_train_new_1 = []
    x_test_new_1 = []
    l = range(1, 2, 1)
    test_data,train_data,whole_data = create_samples(path)
    for i in range(1,2, 1):
        train = d_3_1(train_data, i, 0)
        train_2 = (DataFrame(train)).values
        x_train = np.delete(train_2, train_2.shape[1] - 1, axis=1)
        y_train = train_2[:, train_2.shape[1] - 1]

        test = d_3_1(test_data,i,1)
        test_2= (DataFrame(test)).values
        x_test = np.delete(test_2,test_2.shape[1]-1,axis =1)
        y_test = test_2[:,test_2.shape[1]-1]
        print(train_2.shape)

        result_1, result_2,x_train_new,x_test_new=features_backward_selection(x_train, y_train, x_test, y_test)
        accuracy.append(result_2)
        features.append(result_1)
        x_train_new_1.append(x_train_new)
        x_test_new_1.append(x_test_new)
    accuracy_index = (int)(np.argmax(np.array(accuracy)))
    best = (l[accuracy_index],features[accuracy_index])
    print(best)
    print( 'The best test score is %f'%(test_LogisticRegression(x_train_new_1[accuracy_index], x_test_new_1[accuracy_index], y_train, y_test)))
    print('the wrong way is that just assuming the number of predictors we should use, \n'
          'and then use cross validation to select the best l in this case.\n'
          'the right way is that using cross validation in both step 1 and 2, that is \n'
          'using CV to find the best number of predicators, and then usc CV to find \n'
          'the best number of l')
    return l[accuracy_index],features[accuracy_index]


def dram_diagram(l):
    test_data, train_data, whole_data = create_samples(path)
    train = d_3_1(train_data, l, 0)
    train_2 = (DataFrame(train)).values
    x_train = np.delete(train_2, train_2.shape[1] - 1, axis=1)
    y_train = train_2[:, train_2.shape[1] - 1]

    test = d_3_1(test_data, l, 1)
    test_2 = (DataFrame(test)).values
    x_test = np.delete(test_2, test_2.shape[1] - 1, axis=1)
    y_test = test_2[:, test_2.shape[1] - 1]
    print(train_2.shape)
    result_1, result_2, x_train_new, x_test_new = features_backward_selection(x_train, y_train, x_test, y_test)
    print(result_1,result_2)

    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    predic_train = classifier.predict(x_train)
    predic_test  = classifier.predict(x_test)
    vc_matrix1 = confusion_matrix(y_train, predic_train)
    vc_matrix2 = confusion_matrix(y_test, predic_test)

    predictions = classifier.predict_proba(x_train)  # 每一类的概率.我的概率都是百分百了。。。auc也画不出来啊
    print(predictions)
    false_positive_rate, recall, thresholds = roc_curve(y_train, predictions[:, 1])
    roc_auc = auc(false_positive_rate, recall)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('tpr')
    plt.xlabel('fpr')

    print(vc_matrix1)
    print(vc_matrix2)
    print(classifier.coef_)
    print(classifier.intercept_)
    test_score = classifier.score(x_test, y_test)
    print(test_score)
    plt.show()

def e_1():
    C_1 = []
    train_score_1 = []
    test_score_1 = []
    test_data, train_data, whole_data = create_samples(path)
    l_1 = range(1,21,1)
    for l in range(1,21,1):
        train = d_3_1(train_data, l, 0)
        train_2 = (DataFrame(train)).values
        x_train = np.delete(train_2, train_2.shape[1] - 1, axis=1)
        y_train = train_2[:, train_2.shape[1] - 1]

        test = d_3_1(test_data, l, 1)
        test_2 = (DataFrame(test)).values
        x_test = np.delete(test_2, test_2.shape[1] - 1, axis=1)
        y_test = test_2[:, test_2.shape[1] - 1]
        print(train_2.shape)

        train_set_x = normalize(x_train,axis = 1)
        test_set_x = normalize(x_test,axis = 1)
        cv = StratifiedKFold(n_splits=5)  # stratified method, 5 folds
        classifier_new =LogisticRegressionCV(scoring='accuracy', penalty='l1', solver='liblinear', cv=cv)
        classifier_new.fit(train_set_x, y_train)
        train_set_predic = classifier_new.predict(train_set_x)
        train_score = classifier_new.score(train_set_x, y_train)
        test_score = classifier_new.score(test_set_x, y_test)
        C = classifier_new.C_
        C_1.append(C)
        train_score_1.append(train_score )
        print(train_score )
        print(test_score)
        test_score_1.append(test_score)

    num = (int)(np.argmax(np.array(test_score_1)))
    print('The best C is ', C_1[num])
    print('The bset test accuracy is ', test_score_1[num])
    print('The best l is ',l_1[num])

def f_1(classifier_name='LogisticRegression'):
    C_1 = []
    train_score_1 = []
    test_score_1 = []
    test_sample = []
    train_sample = []
    l_1 = range(1,21,1)
    test_data, train_data, whole_data,y_train,y_test= create_samples_multi(path)
    activity = ['bending1', 'bending2', 'cycling', 'lying', 'sitting', 'standing', 'walking']
    for l in range(1,21,1):
        train = d_3_1(train_data, l, 0)
        train_2 = (DataFrame(train)).values
        x_train_1 = np.delete(train_2, train_2.shape[1] - 1, axis=1)

        test = d_3_1(test_data, l, 1)
        test_2 = (DataFrame(test)).values
        x_test_1 = np.delete(test_2, test_2.shape[1] - 1, axis=1)
        cv = StratifiedKFold(n_splits=5)  # stratified method, 5 folds
        test_sample.append(x_test_1)
        train_sample.append(x_train_1)
        
        if classifier_name == 'LogisticRegression':
            classifier = LogisticRegressionCV(solver='liblinear', penalty='l1', multi_class='ovr',cv=cv)
            x_train_1 = normalize(x_train_1)
            x_test_1  = normalize(x_test_1)
        elif classifier_name == 'GaussianNB':
            classifier = GaussianNB()
        elif classifier_name == 'MultinomialNB':
            classifier = MultinomialNB()
        else:
            pass

        classifier.fit(x_train_1, y_train)
        train_set_predic = classifier.predict(x_train_1)
        test_set_predic = classifier.predict(x_test_1)
        vc_matrix_test = confusion_matrix(y_test, test_set_predic)
        vc_matrix_train = confusion_matrix(y_test, test_set_predic)


        test_score = classifier.score(x_test_1, y_test)

        test_error = 1 - classifier.score(x_test_1, y_test)
        print('Test error = ' + str(test_error))
        test_score_1.append(test_error)
        # print('Following is the confusion matrix of test: ')
        # print(vc_matrix_test)
        # print('Following is the confusion matrix of train: ')
        # print(vc_matrix_train)

    Num = (int)(np.argmin(test_score_1))
    plt.figure()
    xdata = dict()
    ydata = dict()
    x_train_1  = train_sample[Num]
    x_test_1 = test_sample[Num]
    print('The lowest test error is ',test_score_1[Num])
    print('The best l is ', l_1[Num])

    if classifier_name == 'LogisticRegression':
        classifier = LogisticRegressionCV(solver='liblinear', penalty='l1', multi_class='ovr', cv=cv)
        x_train_1 = normalize(x_train_1)
        x_test_1 = normalize(x_test_1)
    elif classifier_name == 'GaussianNB':
        classifier = GaussianNB()
    elif classifier_name == 'MultinomialNB':
        classifier = MultinomialNB()
    else:
        pass
    classifier.fit(x_train_1, y_train)
    test_score = classifier.predict_proba(x_test_1)
    auc_var = dict()
    test_set_y = label_binarize(y_test, classes=activity)
    for i in range(0, 7):
        res = roc_curve(test_set_y[:, i], test_score[:, i])
        xdata[i] = res[0]
        ydata[i] = res[1]
        auc_var[i] = auc(res[0], res[1])
        plt.plot(res[0], res[1], lw=2, label=activity[i] + 'AUC = ' + str(auc_var[i]))
    plt.legend(loc='lower right')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

# test_data, train_data, whole_data = create_samples(path)
# # feature_extraction(whole_data)
# # d_1(train_data)
# # d_2_1(train_data)


# l,num = find_best()
# dram_diagram(l)

# e_1()
f_1(classifier_name='LogisticRegression')
f_1(classifier_name='GaussianNB')
f_1(classifier_name='MultinomialNB')
plt.show()

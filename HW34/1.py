# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LinearRegression
# from sklearn.datasets import load_boston
# import seaborn as sns
# boston=load_boston()
# X=boston["data"]
# Y=boston["target"]
# names=boston["feature_names"]
# print(names)
# print(X)
#
# lr=LinearRegression()
# rfe=RFE(lr,n_features_to_select=12)#选择剔除1个
# rfe.fit(X,Y)
#
# print(rfe.support_) #选中了哪些要删除的对象，没有用的，它已经替我写好了，我就需要这个就够了。
# print(rfe.ranking_)
# print ("features sorted by their rank:")
# print (sorted(zip(map(lambda x:round(x,4), rfe.ranking_),names)))
# # [(1, 'CHAS'), (1, 'CRIM'), (1, 'DIS'), (1, 'LSTAT'), (1, 'NOX'), (1, 'PTRATIO'), (1, 'RAD'), (1, 'RM'), (2, 'INDUS'), (3, 'ZN'), (4, 'TAX'), (5, 'B'), (6, 'AGE')]
# # train = d_3_1(train_data, i, 0)
# # train_2 = (DataFrame(train)).values
# # x_train = np.delete(train_2, train_2.shape[1] - 1, axis=1)
# # y_train = train_2[:, train_2.shape[1] - 1]
# # # print(DataFrame(train))
# # aaa = np.linalg.matrix_rank(x_train)
# # print(aaa, x_train.shape)
# # U, sigma, VT = la.svd(x_train)
# # print(sigma)
# #
# # test = d_3_1(test_data, i, 1)
# # test_2 = (DataFrame(test)).values
# # x_test = np.delete(test_2, test_2.shape[1] - 1, axis=1)
# # y_test = test_2[:, test_2.shape[1] - 1]
# # logit_model = sm.Logit(y_train, x_train)
# # result = logit_model.fit()
# # print(result.summary2())


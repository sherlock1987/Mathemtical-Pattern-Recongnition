import numpy as np
import difflib
import math
import matplotlib.pyplot as plt


# p =[]
# log_p_1 = []
# log_p = range(1,11,1)
# for i in range(len(log_p)):
#     log_p_1.append(log_p[i]/10)
#     p.append(math.pow(10,log_p_1[i]))
# print (p)
#
# plt.figure()
# plt.plot(y, 'b-', linewidth=2)
# plt.legend()
# plt.grid(True)
#
# # 绘制图2
# plt.figure()
# plt.plot(z, 'b-', linewidth=2)
# plt.legend()
# plt.grid(True)
# plt.draw()

a =b = []
a.append(54)
b.append(3232)
print (a,b)

c_3_rate = []
c_3_rate_lowest = []
k_ = range(1, 211, 5)
for i in range(len(n)):
    train, test = train_test_sample(n[i])
    y_train, y_test, train_1, test_1 = data_preprocessing(train, test)
    c_3_rate = []
    for j in range(len(k_)):
        knn = KNeighborsClassifier(n_neighbors=k_[j], metric='euclidean')
        knn.fit(train_1, y_train)
        y_pred = knn.predict(test_1)
        c_3_rate.append(zero_one_loss(y_test, y_pred))
    print('n is %d' % (n[i]))
    c_3_rate_lowest.append(min(c_3_rate))

plt.figure()
plt.title('Learning curve')
plt.xlabel('numbers of train sample')
plt.ylabel('error')
plt.plot(n, c_3_rate_lowest)

# best_n = n[c_3_rate.index(min(c_3_rate))]
# train, test = train_test_sample(best_n)
# y_train, y_test, train_1, test_1 = data_preprocessing(train, test)
# error_lowest, best_k, y_pred_best = find_error(train, test, k_=range(1, 211, 5))
# print ('The best rate is %f in response to number of train sample is %d with k is %d'%(min(c_3_rate),best_n ,best_k))



plt.subplot(321)
sns.boxplot(x = 'class',y = 'PI',hue="class", palette=["m", "g"],
            data=data)
sns.despine(offset=10, trim=True)
plt.subplot(322)
sns.boxplot(x = 'class',y = 'PT',hue="class", palette=["m", "g"],
            data=data)
sns.despine(offset=10, trim=True)
plt.subplot(323)
sns.boxplot(x = 'class',y = 'LLA',hue="class", palette=["m", "g"],
            data=data)
sns.despine(offset=10, trim=True)
plt.subplot(324)
sns.boxplot(x = 'class',y = 'SS',hue="class", palette=["m", "g"],
            data=data)
sns.despine(offset=10, trim=True)
plt.subplot(325)
sns.boxplot(x = 'class',y = 'PR',hue="class", palette=["m", "g"],
            data=data)
sns.despine(offset=10, trim=True)



plt.subplot(326)
sns.boxplot(x = 'class',y = 'GOS',hue="class", palette=["m", "g"],
            data=data)
sns.despine(offset=10, trim=True)


shujufenge

train_AB = data.loc[data['class']=='AB'].head(140)
train_NO = data.loc[data['class']=='NO'].head(70)
train = train_AB.append(train_NO)

test_AB = data.loc[data['class']=='AB'][140:]
test_NO = data.loc[data['class']=='NO'][70:]
test = test_AB.append(test_NO)

print (train)
print (test)

# a = np.where(data1 == 'AB')[0]

#
# # sns.pairplot(data, hue="class",diag_kind="kde")
# plt.show()
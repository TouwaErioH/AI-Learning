import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm

# 加载数据，简单查看信息
def loaddata(file):
    csv_data=pd.read_csv(file, encoding='gb18030') # 编码
    csv_data.head()  # 显示前5行数据
    csv_data.info()  # 查看各字段的信息
    csv_data.shape  # 查看数据集行列分布，几行几列
    csv_data.describe()  # 查看数据的大体情况
    return csv_data

# 对原始数据简单的分析
def dataanalyse(csv_data):
    #csv_data.corr('pearson')
    label = csv_data['事件类型']  # Name: 事件类型, Length: 99809, dtype: object
    uni = label.unique()  # <class 'numpy.ndarray'>
    print(uni)
    print(len(uni))

    #图形化展示label分布
    df = csv_data
    d = {'label': df['事件类型'].value_counts().index, 'count': df['事件类型'].value_counts()}
    df_label = pd.DataFrame(data=d).reset_index(drop=True)
    df_label.plot(x='label', y='count', kind='bar', legend=False, figsize=(8, 5))
    plt.title("labels")
    plt.ylabel('count', fontsize=18)
    plt.xlabel('label', fontsize=18)
    plt.show()

    print(csv_data['目的端口'].value_counts())  # 统计某一列中各个元素值出现的次数
    print("Skewness: %f" % (csv_data['目的端口'].skew()))  # 列出数据的偏斜度
    print("Kurtosis: %f" % (csv_data['目的端口'].kurt()))  # 列出数据的峰度
    # for index, row in csv_data.iteritems():
    #     print(csv_data[index].value_counts())  # 统计某一列中各个元素值出现的次数
    #     print("Skewness: %f" % (csv_data[index].skew()))  # 列出数据的偏斜度
    #     print("Kurtosis: %f" % (csv_data[index].kurt()))  # 列出数据的峰度

# 缺失数据查看，处理（填充）
def data_null(csv_data):
    #查看缺失值，含有缺失值的特征，特征含有缺失值的数量
    print(csv_data.isnull())
    missing = csv_data.columns[csv_data.isnull().any()].tolist()
    print(missing)
    print(csv_data[missing].isnull().sum())

    #含有缺失值的特征的类型
    for index, row in csv_data.iteritems():
        a = csv_data[index]
        if a.isnull().sum() > 0:
            print(index)
            print(a.dtype)
    #缺失值填充
    data = csv_data
    temp = data
    for index, row in data.iteritems():
        if temp[index].dtype == object:
            temp[index] = temp[index].fillna('0')
        else:
            temp[index] = temp[index].fillna(0)
    print(temp)
    #判断得到temp中"看不到"的数据为空格
    if temp['源IP地址'][4] == ' ':
        print("x")
    if temp['备用字符串7'][2] == ' ':
        print("y")
    return temp,data

#尝试onehot编码，超出内存限制
def try_onehot(csv_data):
    pdata = csv_data.drop(['对象', '持续时间', '设备IP地址', '事件接收时间', '源IP地址', '目的IP地址', '采集器IP地址',
                           '事件原始内容', '事件内容摘要', '事件产生时间'], axis=1)
    ptemp = pdata
    for index, row in pdata.iteritems():
        if ptemp[index].dtype == object:
            ptemp[index] = ptemp[index].fillna('0')
        else:
            ptemp[index] = ptemp[index].fillna(0)
    enc = preprocessing.OneHotEncoder(categorical_features='all')
    enc.fit(ptemp)
    ans = enc.transform(ptemp).toarray()  # 直接  toarray()   MemoryError:   99809x29205

# 顺序编码，factorize  计算数据相关性
def encod(temp,data):
    for index, row in data.iteritems():
        if temp[index].dtype == object:
            temp[index] = pd.factorize(data[index])[0].astype(np.uint16)
    print(temp)

    #为了方便处理，都转化为 float 转化后 各列特征，均为数字。
    # 计算各列相关性 由于某列各值都相同的话，编码都为0，相关性计算无法计算，为NaN，先处理掉
    df = temp.astype("float64")
    tp = df
    for index, row in tp.iteritems():
        if temp[index].value_counts().shape[0] == 1:  # if temp[index].sum()==0: 优化
            print(index)
            df = df.drop([index], axis=1)

    #计算相关性
    tp=df.corr("pearson")
    print(tp)
    print(tp.isnull().any())
    return df,tp

#数据标准化
# def data_normalizing():
#     # 这里先不采用。
#     # 原因：比如逻辑回归中采取下列标准化，准确率召回率低于
#     # 10 %
#     # sc = StandardScaler()
#     # sc.fit(x_train)  # 计算均值和方差
#     # x_train_std = sc.transform(x_train)  # 利用计算好的方差和均值进行Z分数标准化
#     # x_test_std = sc.transform(x_test)  # 这里不合适。采用了之后准确率召回率低于 10%

#特征选择
# 首先排除低方差的特征，然后对剩余特征进行pearson相关系数计算，选择绝对值较高的特征。
# 同时，由于pearson相关系数只对于线性关系敏感，可能具有误导性，故在排除低方差特征后也进行卡方检验，
# 计算特征和label的相关性，选择最优的几个特征，然后与根据相关系数选择出的特征进行对比。
# 除数学计算外，也根据人为经验选择一些对分类重要影响的特征。
# 对上述得到的三组特征分别进行模型训练，根据预测效果评分，选择合适的特征。
# 最后进行特征的组合，筛选（单独增减某些特征，对比预测效果）。
# 此过程对于下面的四种模型（SVM,朴素贝叶斯，逻辑回归，神经网络）分别独立，
# 也就是进行四轮，每轮中从原始的三组特征开始不断调整，得到多组特征，对比预测效果，从而针对每个模型得到最合适的特征。
def feature_selc(feature,label,tp):
    #根据相关性排序选择特征
    print(tp['事件类型'].sort_values())

    # 根据卡方选择相关性高的k个特征
    X = feature
    y = label
    X_new = SelectKBest(chi2, k=6).fit_transform(X, y)
    # 原来的特征名称与选择出的对应
    X_n = X.values
    for i in range(46):
        for j in range(6):
            if (X_n[:, i:i + 1] == X_new[:, j:j + 1]).all():
                print('%d %d' % (i, j))
                print(X.iloc[:, i].name)

#模型评估
# def judge(predictions):
#     print(precision_score(y_test, predictions, average='weighted'))
#     print('-========')
#     print(recall_score(y_test, predictions, average='weighted'))
#     print('-========')
#     print(classification_report(y_test, predictions))
#     from sklearn.metrics import confusion_matrix
#     print(confusion_matrix(y_test, predictions))
#     print(type(classification_report(y_test, predictions)))

#k折交叉验证
# def kjudge():
#     from sklearn.model_selection import RepeatedKFold
#     kf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=0)  # 3次  2折 交叉验证
#     for train_index, test_index in kf.split(X):
#         print('train_index', train_index, 'test_index', test_index)
#         train_X, train_y = X[train_index], y[train_index]
#         test_X, test_y = X[test_index], y[test_index]
#         lr = LR()
#         lr.fit(train_X, train_y)
#         predictions = lr.predict(test_X)
#         from sklearn.metrics import precision_score, recall_score
#         from sklearn.metrics import classification_report
#         print(precision_score(test_y, predictions, average='micro'))
#         print('-========')
#         print(recall_score(test_y, predictions, average='micro'))
#         print('-========')
#         print(classification_report(test_y, predictions))

# 对选择好的特征PCA处理，测试后发现效果不理想，未采用
# def datapca():
#     from sklearn.model_selection import train_test_split
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=3)
#     pca.fit(X)
#     np.set_printoptions(precision=None,suppress=False)
#     print(pca.explained_variance_ratio_)   #降维后的各主成分的方差值占总方差值的比例
#     print(pca.explained_variance_)  # 降维后的各主成分的方差值
#     X = pca.transform(X)

# 逻辑回归
def logr(feature,label):
    lin = feature[['操作', '备用字符串2', '事件名称', '事件原始等级', '事件原始类型', '目的端口', '结果',
                   '网络协议', '网络应用协议',
                   '事件内容摘要', '备用字符串1', '设备IP地址', '响应']]
    X = lin
    y = label
    y = label.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
    lr = LR(penalty='l2', C=1.0, solver='liblinear', class_weight=None, max_iter=100, tol=0.0001)
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)

    print(precision_score(y_test, predictions, average='weighted'))
    print('-========')
    print(recall_score(y_test, predictions, average='weighted'))
    print('-========')
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(type(classification_report(y_test, predictions)))

def gsN(feature, label):
    lin = feature[['事件原始类型', '事件原始等级', '事件名称', '备用字符串2', '操作', '备用字符串1', '结果',
                   '响应', '设备IP地址', '事件内容摘要'
                   ]]
    X = lin
    y = label
    y = label.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
    lr = GaussianNB()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)

    print(precision_score(y_test, predictions, average='weighted'))
    print('-========')
    print(recall_score(y_test, predictions, average='weighted'))
    print('-========')
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(type(classification_report(y_test, predictions)))


def MLP(feature, label):
    import numpy as np
    lin = feature[['事件原始类型', '事件原始等级', '事件名称', '备用字符串2', '操作', '备用字符串1',
                   '结果', '响应', '设备IP地址', '网络协议']]
    X = lin
    y = label
    y = label.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                        alpha=0.0001, max_iter=200, tol=0.0001)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print(precision_score(y_test, predictions, average='weighted'))
    print('-========')
    print(recall_score(y_test, predictions, average='weighted'))
    print('-========')
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(type(classification_report(y_test, predictions)))

def mySVM(feature, label):
    lin = feature[['事件原始类型', '事件原始等级', '事件名称', '备用字符串2',
                   '操作', '备用字符串1', '结果', '响应', '设备IP地址', '网络协议']]
    X = lin
    y = label
    y = label.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
    lr = svm.LinearSVC(penalty='l2', loss='squared_hinge', tol=0.0001, C=1.0, multi_class='ovr',
                       class_weight=None, max_iter=1000)  # SVC() 复杂度 n^2太高
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)

    print(precision_score(y_test, predictions, average='weighted'))
    print('-========')
    print(recall_score(y_test, predictions, average='weighted'))
    print('-========')
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(type(classification_report(y_test, predictions)))

if __name__=='__main__':
    file='./event.csv'
    csv_data=loaddata(file)
    dataanalyse(csv_data)
    temp,data=data_null(csv_data)
    #try_onehot(csv_data)
    df,tp=encod(temp,data)

    # 根据编码前后的label数量可以一一对应
    # csv_data['事件类型'].value_counts()
    # df['事件类型'].value_counts()

    #data_normalizing()
    label = df['事件类型']
    label = label.to_frame()
    feature = df.drop(['事件类型'], axis=1)
    feature_selc(feature,label,tp)

    #特征选择过程见文档说明及pre.ipynb，下面的四个模型采用已经选择好的最优特征
    logr(feature, label)
    gsN(feature, label)
    MLP(feature, label)
    mySVM(feature, label)

    #kjudge() 可选

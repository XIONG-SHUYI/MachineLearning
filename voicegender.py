import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

def load_data():
    voice_data=pd.read_csv('D:\\voice.csv')
    x=voice_data.iloc[:,:-1]
    y=voice_data.iloc[:,-1]
    y = LabelEncoder().fit_transform(y)
    imp=SimpleImputer(missing_values=0, strategy='mean')
    x=imp.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    scaler1 = StandardScaler()
    scaler1.fit(x_train)
    x_train = scaler1.transform(x_train)
    x_test = scaler1.transform(x_test)
    return x_train, x_test, y_train, y_test

class NaiveBayes:
    def __init__(self):
        self.model = None
    # 数学期望
    @staticmethod  # 在def方法中没有引用对象的资源时，要加上这个
    def mean(X):  # 这表明mean这个函数里面不需要调用self.的对象和调用
        return sum(X) / float(len(X))
    # 标准差（方差） X是所有样本同一特征下的各个值
    def stdev(self, X):  # 这里就不能加@staticmethod，因为他要用刚才定义好的函数mean
        avg = self.mean(X)  # 依然是用self来引用函数
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))
    # 概率密度函数  x结构为一个数据，其格式为[第一个特征, 第二个特征]
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):  # train_data结构为[array[5.0, 0.37],array[3.42, 0.40],array[3.42, 0.40],[12, 0.40]..]
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]  # zip(*)作用将train_data转置，
        # 变为 [array[5.0, 3.42, 12], array[0.37, 0.4, 0.4]]
        return summaries

    # 分类别求出数学期望和标准差    self.model的数据结构{类别1：第一个特征期望和方差，第二个特征的期望和方差
    #                                                    类别2：第一个特征期望和方差，第二个特征的期望和方差}
    def fit(self, X, y):
        labels = list(set(y))  # 集合是{} 列表是[] 所以要list回来
        data = {label: [] for label in labels}  # 创建个字典，每个键的值是列表
        for f, label in zip(X, y):  # zip将对象相应位置元素打包成元组，然后返回元组组成的列表，结构为(array([X]), y),...
            data[label].append(f)  # 构建data{y1:[array(x1),array(x2),..],y2:[array(x1),array(x2),..]..}
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return 'gaussianNB train done!'

    # 计算概率 probabilities的结构为{0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
    def calculate_probabilities(self, input_data):
        # input_data[i]:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    # 选出概率大所对应的类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(self.calculate_probabilities(X_test).items(),key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        fright=0
        mright=0
        mnum=0
        fnum=0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if y==1:
                mnum+=1
                if label == y:
                    right += 1
                    mright+=1
            if y==0:
                fnum+=1
                if label == y:
                    right += 1
                    fright+=1
        return right / float(len(X_test)), mright / float(mnum), fright / float(fnum)

X_train, X_test, y_train, y_test=load_data()
model = NaiveBayes()
model.fit(X_train, y_train)
rate, mrate, frate=model.score(X_test, y_test)
print("男声正确率:", mrate, "男声错误率", 1-mrate, "女声正确率:", frate, "女声错误率", 1-frate)

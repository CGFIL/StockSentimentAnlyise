import re
import numpy as np

class Data(object):
    def __init__(self, path="D:/gdy/SentimentAnalyze/data/"):
        '''
        根据路径读取csv文件创建data对象
        :param path: 训练集路径
        '''
        self.negative_data = [re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[\[\]+——！，。？、~@#￥%……&*（）]+", "", \
                       line[:-1]) for line in open(path + 'negative.txt',encoding="utf_8")]

        self.neural_data = [re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[\[\]+——！，。？、~@#￥%……&*（）]+", "", \
                       line[:-1]) for line in open(path + 'neural.txt',encoding="utf_8")]

        self.positive_data = [re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[\[\]+——！，。？、~@#￥%……&*（）]+", "", \
                       line[:-1]) for line in open(path + 'positive.txt',encoding="utf_8")]


    def create_classs(self):
        '''
        根据数据集创建对应类型
        :return: 类型数组
        '''

        num_positive = len(self.positive_data)
        num_negative = len(self.negative_data)
        num_neural = len(self.neural_data)

        data_labels = np.zeros([num_positive+num_negative+num_neural,3])
        data_labels[:num_negative] = [1, 0, 0]
        data_labels[num_negative:num_neural + num_negative] = [0, 1, 0]
        data_labels[-num_positive:] = [0, 0, 1]

        return data_labels
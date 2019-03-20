import re
import numpy as np

class Data(object):
    def __init__(self, path="/home/jack/PycharmProjects/StockSentimentAnlyise/data/"):
        '''
        根据路径读取csv文件创建data对象
        :param path: 训练集路径
        '''
        self.negative_data = [re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[\[\]+——！，。？、~@#￥%……&*（）]+", "", \
                       line[:-1]) for line in open(path + 'negative.txt')]

        self.neural_data = [re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[\[\]+——！，。？、~@#￥%……&*（）]+", "", \
                       line[:-1]) for line in open(path + 'neural.txt')]

        self.positive_data = [re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[\[\]+——！，。？、~@#￥%……&*（）]+", "", \
                       line[:-1]) for line in open(path + 'positive.txt')]


    def create_classs(self):
        '''
        根据数据集创建对应类型
        :return: 类型数组
        '''

        num_positive = len(self.positive_data)
        num_negtive = len(self.negative_data)
        num_neural = len(self.neural_data)

        return np.concatenate((np.full((1, num_negtive), -1)[0] ,np.zeros(num_neural), np.ones(num_positive)))
import csv

class Data(object):
    def __init__(self, path="/home/jack/PycharmProjects/StockSentimentAnlyise/data/sample1.CSV"):
        '''
        根据路径读取csv文件创建data对象
        :param path: 训练集路径
        '''
        self.data = open(path, newline='')


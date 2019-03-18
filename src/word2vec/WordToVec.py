import re
import third.thulac as th

class Word2Vec(object):
    def __init__(self, path="/home/jack/PycharmProjects/StockSentimentAnlyise/third/models"):
        '''
        WordToVec
        :param path: 存放模型路径
        '''
        self.thu_model = th.thulac(model_path=path, seg_only=True)
        return

    @staticmethod
    def clear_punctuation(sentence):
        '''
        去除句子中的特殊符号
        :param sentence:待处理句子
        :return: srting
        '''
        return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", sentence)

    def parse_text_to_words(self, sentence):
        '''
        将对句子进行分词，并返回分词后的列表
        :param sentence: 待分词语句
        :return: []
        '''
        return self.thu_model.cut(sentence, text=True)
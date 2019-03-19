import re
import third.thulac as th
from gensim.models import KeyedVectors

class Word2Vec(object):
    def __init__(self, path="/home/jack/PycharmProjects/StockSentimentAnlyise/third/models"):
        '''
        WordToVec
        :param path: 存放模型路径
        '''
        self.thu_model = th.thulac(model_path=path, seg_only=True)
        self.w2v_model = KeyedVectors.load_word2vec_format(path +  "/sgns.financial.bigram", binary=False)

        return

    def __clear_punctuation(self, sentence):
        '''
        去除句子中的特殊符号
        :param sentence:待处理句子
        :return: srting
        '''
        return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", sentence)

    def __parse_text_to_words(self, sentence):
        '''
        将对句子进行分词，并返回分词后的列表
        :param sentence: 待分词语句
        :return: []
        '''
        words = self.thu_model.cut(self.__clear_punctuation(sentence));
        return [word[0] for word in words]

    def parse_word_to_vector(self, word):
        '''
        将句子中的单个词转换为预训练模型中向量
        :param word: 词
        :return: 向量
        '''
        print(word)
        vec = self.w2v_model[word]
        return vec

    def parse_sentence_to_vectors(self, sentence):
        '''
        将句子转换为单词后返回每个词的预训练模型向量
        :param sentence:
        :return:
        '''
        words = self.__parse_text_to_words(sentence)
        print(words)
        vectors = [self.parse_word_to_vector(word) for word in words]
        return vectors

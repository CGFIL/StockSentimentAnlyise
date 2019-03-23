import re
import numpy as np
from src.word2vec.WordToVec import Word2Vec
from src.training.traininig_data import Data
import third.thulac as th

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


def test():
    #测试模型加载
    #print(w2v.w2v_model.similarity("银行", "券商"))
    w2v = Word2Vec("../third/models")
    #print(type(w2v.thu_model.cut("银行券商华为中兴")))

    #测试训练数据集
    '''
    数据集难处理
    '''
    #data = Data(path="../data/sample1.CSV")
    #print(data)

    #测试向量生成
    for vec in w2v.parse_sentence_to_vectors("大盘重返3100，提前发来贺信"):
        print(vec)

def test_thu(sentence):
    thu_model = th.thulac(model_path="../third/models", seg_only=True)
    print([word[0] for word in thu_model.cut(sentence)])

def test_data():
    data = Data(path="../data/")
    postitve_data = [re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[\[\]+——！，。？、~@#￥%……&*（）]+", "", line[:-1]) for line in data.positive_data]
    for item in postitve_data:
        test_thu(item)

def test_seq():
    w2v = Word2Vec(len=50, path="../third/models")
    data = Data("../data/")
    nag_text = [re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[\[\]+——！，。？、~@#￥%……&*（）]+", "", \
                       line[:-1]) for line in data.negative_data]

    print(nag_text)
    seqcences = w2v.convert_sentences_to_seqences(nag_text)
    print(seqcences)

def test_class():
    data = Data(path="../data/")
    traning_res = data.create_classs()
    print(traning_res)


def traning():
    '''
    训练函数
    :return:
    '''
    num_of_word = 50
    words_use   = 40000

    #初始化模型并加载
    data = Data(path="../data/")
    w2v  = Word2Vec(len=num_of_word, path="../third/models")

    #构建训练数据集和训练结果集
    sentences = data.negative_data + data.neural_data+ data.positive_data
    traning_seq = w2v.convert_sentences_to_seqences(sentences)
    traning_res = data.create_classs()

    print(traning_seq)
    print(traning_res)

    #创建输入层向量
    embding_dimen = w2v.w2v_model["银行"].shape[0]
    embding_mat   = np.zeros((words_use, embding_dimen))
    for i in range(num_of_word):
        embding_mat[i, :] = w2v.w2v_model[w2v.w2v_model.index2word[i]]
    embding_mat = embding_mat.astype('float32')

    traning_seq[traning_seq >= words_use] = 0


    #创建模型
    model = Sequential()
    model.add(Embedding(words_use, embding_dimen, weights=[embding_mat], input_length=num_of_word, trainable=False))

    model.add(GRU(units=32, return_sequences=True))
    model.add(GRU(units=16, return_sequences=False))
    model.add(Dense(3, activation='softmax'))


    #模型编译
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    #模型训练 使用后10个数据测试
    model.fit(traning_seq[:-10], traning_res[:-10],
              validation_split=0.1,
              epochs=40,
              batch_size=128)

    #使用后10个数据测试
    result = model.evaluate(traning_seq[-10:], traning_res[-10:])
    print('Accuracy:{0:.3%}'.format(result[1]))


if __name__ == '__main__':
    traning()


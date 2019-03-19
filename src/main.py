from src.word2vec.WordToVec import Word2Vec
from src.training.traininig_data import Data
import third.thulac as th

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
    for vec in w2v.parse_sentence_to_vectors("银行券商华为中兴"):
        print(vec)

def test_thu():
    thu_model = th.thulac(model_path="../third/models", seg_only=True)
    print([word[0] for word in thu_model.cut("银行券商华为中兴")])

if __name__ == '__main__':
    test()

from src.word2vec.WordToVec import Word2Vec

w2v = Word2Vec("../third/models")


def convert_sentence2vec(sentence):
    return w2v.parse_text_to_words(sentence)


if __name__ == '__main__':
    test_sen = "华为怎么可能和京东方合作,他们订购了一批三星显示屏"
    test_sen = Word2Vec.clear_punctuation(test_sen)
    word_lists = convert_sentence2vec(test_sen)
    print(word_lists)
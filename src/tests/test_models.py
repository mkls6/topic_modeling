"""
Test implemented topic models
"""

from models import LDA, Top2VecW
from preproc import Preprocessor, LangEnum
from defines import (TARGET_TEXTS_EN,
                     COMPARISON_TOPICS_EN,
                     TARGET_TEXTS_RU,
                     COMPARISON_TOPICS_RU)

# Preprocessor setup
EN_PREPROCESSOR = Preprocessor(LangEnum.EN)
TEXTS, DICTIONARY = EN_PREPROCESSOR.preprocess_texts(TARGET_TEXTS_EN)
DOC2BOW_CORPUS = [DICTIONARY.doc2bow(doc)
                  for doc in map(lambda x: [y.lower_ for y in x], TEXTS)]

# LDA variants
LDA_MODEL = LDA(id2word=DICTIONARY, random_state=42)
LDA_MULTICORE = LDA(workers=6, id2word=DICTIONARY, random_state=42)

# NMF
NMF_MODEL = None

# Top2Vec variants
TOP2VEC_D2V_MODEL = None
TOP2VEC_BERT_MODEL = None


# Single-core LDA model
def test_lda_init():
    topics = list(LDA_MODEL.get_topics(num_topics=1))

    assert topics[0][0] == COMPARISON_TOPICS_EN[0][0]
    assert {x for x, y in topics[0][1]} == COMPARISON_TOPICS_EN[0][1]


def test_lda_update():
    LDA_MODEL.update(DOC2BOW_CORPUS)
    topics = list(LDA_MODEL.get_topics(num_topics=1))

    assert topics[0][0] == COMPARISON_TOPICS_EN[1][0]
    assert {x for x, _ in topics[0][1]} == COMPARISON_TOPICS_EN[1][1]


# Multicore LDA model
def test_lda_multicore():
    topics = list(LDA_MULTICORE.get_topics(num_topics=1))

    assert topics[0][0] == COMPARISON_TOPICS_EN[0][0]
    assert {x for x, y in topics[0][1]} == COMPARISON_TOPICS_EN[0][1]


def test_lda_multicore_update():
    LDA_MULTICORE.update(DOC2BOW_CORPUS)
    topics = list(LDA_MULTICORE.get_topics(num_topics=1))

    assert topics[0][0] == COMPARISON_TOPICS_EN[2][0]
    assert {x for x, _ in topics[0][1]} == COMPARISON_TOPICS_EN[2][1]


# Doc2Vec top2vec variant
def test_top2vec_doc2vec_init():
    global TOP2VEC_D2V_MODEL
    TOP2VEC_D2V_MODEL = Top2VecW(documents=TARGET_TEXTS_RU,
                                 workers=4,
                                 embedding_model='doc2vec')
    topic_nums, words, scores = TOP2VEC_D2V_MODEL.get_topics()
    assert words[0] == COMPARISON_TOPICS_RU[0][0]


def test_top2vec_update():
    pass

# TODO: NMF model

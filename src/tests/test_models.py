"""
Test implemented topic models
"""

from models import LDA
from preproc import Preprocessor, LangEnum
from defines import (TARGET_TEXTS_EN,
                     COMPARISON_SETS_EN,
                     COMPARISON_TOPICS_EN)

EN_PREPROCESSOR = Preprocessor(LangEnum.EN)
TEXTS, DICTIONARY = EN_PREPROCESSOR.preprocess_texts(TARGET_TEXTS_EN)
DOC2BOW_CORPUS = [DICTIONARY.doc2bow(doc) for doc in TEXTS]

LDA_MODEL = LDA(id2word=DICTIONARY, random_state=42)
LDA_MULTICORE = LDA(workers=6, id2word=DICTIONARY, random_state=42)
NMF_MODEL = None


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

# TODO: NMF model

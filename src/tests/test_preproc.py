from defines import (TARGET_TEXTS_EN,
                     COMPARISON_SETS_EN,
                     TARGET_TEXTS_RU,
                     COMPARISON_SETS_RU)
from preproc import Preprocessor, LangEnum

EN_PREPROCESSOR = Preprocessor(language=LangEnum.EN, stop_words=[])
RU_PREPROCESSOR = Preprocessor(language=LangEnum.RU, stop_words=[])


def test_english_preprocessing():
    texts, dictionary = EN_PREPROCESSOR.preprocess_texts(TARGET_TEXTS_EN)

    assert {token.lower_ for text in texts for token in text} == \
           COMPARISON_SETS_EN[0]
    assert set(dictionary.values()) == COMPARISON_SETS_EN[0]


def test_russian_preprocessing():
    texts, dictionary = RU_PREPROCESSOR.preprocess_texts(TARGET_TEXTS_RU)

    assert {token.lower_ for text in texts for token in text} == \
           COMPARISON_SETS_RU[0]
    assert set(dictionary.values()) == COMPARISON_SETS_RU[0]


def test_stop_words():
    EN_PREPROCESSOR.update_stopwords(['robot', 'bloop'])
    _, dictionary = EN_PREPROCESSOR.preprocess_texts(TARGET_TEXTS_EN)

    assert set(dictionary.values()) == COMPARISON_SETS_EN[1]

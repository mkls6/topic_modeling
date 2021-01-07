from preproc import Preprocessor, LangEnum

TARGET_TEXTS_EN = [
    'Bleep-bloop, I am a robot!',
    'There is a number (12345) and an email (hello-there@box.com).'
]

COMPARISON_SETS_EN = [
    {'numb', 'robot', 'bloop', 'email', 'bleep'},
    {'numb', 'email', 'bleep'}
]

EN_PREPROCESSOR = Preprocessor(language=LangEnum.EN, stop_words=[])


def test_english_preprocessing():
    _, dictionary = EN_PREPROCESSOR.preprocess_texts(TARGET_TEXTS_EN)

    assert set(dictionary.values()) == COMPARISON_SETS_EN[0]


def test_stop_words():
    EN_PREPROCESSOR.update_stopwords(['robot', 'bloop'])
    _, dictionary = EN_PREPROCESSOR.preprocess_texts(TARGET_TEXTS_EN)

    assert set(dictionary.values()) == COMPARISON_SETS_EN[1]


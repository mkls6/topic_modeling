import spacy
from spacy.language import Language
from spacy.analysis import Doc
from gensim.corpora.dictionary import Dictionary
from enum import Enum
from typing import Iterable


class LangEnum(Enum):
    """
    Enum to represent supported language codes
    """
    EN = 0
    RU = 1


class Preprocessor:
    """
    Use this class to encapsulate Spacy models, Gensim stuff and everything
    else needed for text preprocessing.
    """

    def __init__(self, language: LangEnum = 0,
                 stop_words: Iterable[str] = None):
        # Preload ready to use spacy language model (tokenizer, lemmatizer, etc)
        if language == LangEnum.EN:
            self.nlp: Language = spacy.load('en_core_web_sm',
                                            disable=["tagger",
                                                     "parser",
                                                     "ner"])
        elif language == LangEnum.RU:
            # TODO: load spacy model for russian language
            raise NotImplementedError('Unavailable yet! :(')
        else:
            raise NotImplementedError('Only russian and english '
                                      'languages are supported at the moment')

        # Update the built-in stopwords list
        if stop_words is not None:
            self.update_stopwords(stop_words)

        # Add stop words removing to spacy pipeline
        self.nlp.add_pipe(
            lambda doc: [token.lemma_.lower() for token in doc
                         if not (token.is_stop or
                                 token.is_punct or
                                 token.like_email or
                                 token.like_url or
                                 token.is_space or
                                 token.like_num or
                                 token.lemma_.lower() in
                                 self.nlp.Defaults.stop_words)],
            last=True)

    def update_stopwords(self, stop_words: Iterable[str]) -> None:
        """
        Update built-in spacy language model stopwords list

        :param stop_words: Iterable of strings - target stopwords
        :return: None
        """
        self.nlp.Defaults.stop_words.update(stop_words)
        for word in self.nlp.Defaults.stop_words:
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True

    def preprocess_texts(self,
                         data: Iterable[str]) -> (Iterable[Doc], Dictionary):
        """
        Get preprocessed texts

        :param data: iterable of strings
                     (each string is considered to be a single document)
        :return: preprocessed documents and
                a gensim Dictionary of the given docs
        """
        docs = self.__get_preprocessed_docs__(data)
        return docs, Dictionary(docs)

    def __get_preprocessed_docs__(self,
                                  data: Iterable[str]):
        """
        Helper function to generate new docs using spacy Language.pipe()

        :param data: iterable of strings (1 string = 1 doc)
        :return: spacy Document generator
        """
        docs = self.nlp.pipe(data)
        for doc in docs:
            yield doc

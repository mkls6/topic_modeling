"""
This module contains generic model wrapper and some
default implementations (LDA, NMF).
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable, Tuple


class GenericModel(ABC):
    """
    GenericModel class defines a basic interface
    for all other models. You need to wrap up your model (
    e.g. a gensim model) in such class and implement 2 methods:

    1. fit() which gets data for model training/fitting.
    2. update() to update the model.
    3. get_topics() to return the extracted topics.
    """

    @abstractmethod
    def fit(self, data: Any, *args, **kwargs) -> None:
        """
        Fit freshly created model instance.

        :param data: train data (in any format supported by your model)
        :param args: additional positional args that you need
        :param kwargs: keyword args for your model
        :return: None
        """
        pass

    @abstractmethod
    def update(self, data: Any, *args, **kwargs) -> None:
        """
        Update the model with new data

        :param data: new data to re-fit on
        :param args: additional args if you need this
        :param kwargs: additional kwargs if you need this
        :return: None
        """
        pass

    @abstractmethod
    def get_topics(self,
                   docs: Optional[Any] = None,
                   *args, **kwargs) -> Iterable[Tuple[int, Tuple[str, float]]]:
        """
        Get topics extracted from docs

        :param docs: new document collection,
                     if None (default) returns topics extracted from the
                     latest model state
        :param args: additional positional arguments
        :param kwargs: additional keyword arguments
        :return: topics - any Iterable of Tuple[id, Tuple[word, prob]]
        """
        pass


class LDA(GenericModel):
    """
    Wrapper for Gensim LdaModel and LdaMulticore
    """

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, data: Any, *args, **kwargs):
        pass

    def update(self, data: Any, *args, **kwargs):
        pass

    def get_topics(self, docs: Optional[Any] = None, *args, **kwargs):
        pass


class NMF(GenericModel):
    """
    Wrapper for Gensim Non-negative matrix factorization topic model
    """

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, data: Any, *args, **kwargs):
        pass

    def update(self, data: Any, *args, **kwargs):
        pass

    def get_topics(self, docs: Optional[Any] = None, *args, **kwargs):
        pass

"""
This module contains generic model wrapper and some
default implementations (LDA, NMF).
"""
from abc import ABC, abstractmethod
from typing import Any, Optional


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
    def fit(self, data: Any, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, data: Any, *args, **kwargs):
        pass

    @abstractmethod
    def get_topics(self, doc: Optional = None, *args, **kwargs):
        pass


class LDA(GenericModel):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, data: Any, *args, **kwargs):
        pass

    def update(self, data: Any, *args, **kwargs):
        pass

    def get_topics(self, doc: Optional = None, *args, **kwargs):
        pass


class NMF(GenericModel):
    def fit(self, data: Any, *args, **kwargs):
        pass

    def update(self, data: Any, *args, **kwargs):
        pass

    def get_topics(self, doc: Optional = None, *args, **kwargs):
        pass

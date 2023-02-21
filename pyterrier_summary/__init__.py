import pyterrier as pt
from abc import abstractmethod

class Summarizer(pt.Transformer):
    def __init__(self, body_attr, out_attr, verbose) -> None:
        self.body_attr = body_attr
        self.out_attr = out_attr
        self.verbose = verbose
        
    @abstractmethod
    def transform(self, inp):
        raise NotImplementedError


import pyterrier as pt
from pyterrier.transformer import TransformerBase
import torch
from abc import abstractmethod

class Summarizer(pt.Transformer):
    def __init__(self, body_attr, out_attr, verbose) -> None:
        self.body_attr = body_attr
        self.out_attr = out_attr
        self.verbose = verbose
        
    @abstractmethod
    def transform(self, inp):
        raise NotImplementedError

class NeuralSummarizer(Summarizer, TransformerBase):
    def __init__(self, 
                 model_name, 
                 tokenizer_name=None, 
                 batch_size=32, 
                 device=None, 
                 enc_max_length=180, 
                 body_attr='text', 
                 out_attr='summary', 
                 verbose=False) -> None:
        Summarizer.__init__(self, body_attr, out_attr, verbose)
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.enc_max_length = enc_max_length
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)        


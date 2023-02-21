import pyterrier as pt
from more_itertools import chunked
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from abc import abstractmethod
import logging

class Summarizer(pt.Transformer):
    def __init__(self, body_attr, out_attr, verbose) -> None:
        self.body_attr = body_attr
        self.out_attr = out_attr
        self.verbose = verbose
        
    @abstractmethod
    def transform(self, inp):
        raise NotImplementedError

class NeuralSummarizer(Summarizer):
    def __init__(self, model_name, tokenizer_name=None, mode='inplace', batch_size=4, device=None, enc_max_length=180, body_attr='text', out_attr='summary', verbose=False) -> None:
        super().__init__(body_attr, out_attr, verbose)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.enc_max_length = enc_max_length

        tokenizer_name = tokenizer_name if tokenizer_name else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
    
    def _set_mode(self, inp):
        if self.mode == 'inplace': inp 
        elif self.mode == 'docs': return inp[["docno", self.body_attr]].drop_duplicates(subset="docno")
        else:
            logging.warning(f'Setting {self.mode} not recognised, defaulting to in-place')
            return inp 
        
    def _summarize(self, text):
        results = []
        with torch.no_grad():
            for chunk in chunked(text.tolist(), self.batch_size):
                inps = self.tokenizer(chunk, padding="max_length", truncation=True, max_length=self.enc_max_length, return_tensors="pt")
                inps = inps.input_ids.to(self.device)
                attention_mask = inps.attention_mask.to(self.device)
                out = self.model.generate(inps, attention_mask=attention_mask)
                results.extend(self.tokenizer.batch_decode(out, skip_special_tokens=True))
        return results
    
    @classmethod
    def from_pretrained(cls, model_name, batch_size=32, text_field='text', verbose=False, device=None):
        res = cls(model_name, batch_size=batch_size, device=device, body_attr=text_field, verbose=verbose)
        res.model_name = model_name
        return res
    
    def transform(self, inp):
        assert "docno" in inp.columns and self.body_attr in inp.columns, "Malformed Frame, Expecting Documents"
        out = self._set_mode(inp)
        out[self.out_attr] = self._summarize(out[self.body_attr])
        return out 

        


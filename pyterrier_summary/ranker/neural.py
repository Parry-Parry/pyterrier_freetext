from typing import List
import numpy as np
import pandas as pd
import torch
from more_itertools import chunked
import logging 
from pyterrier_t5 import MonoT5Ranker

from . import split_into_sentences
from .. import NeuralSummarizer

class T5Ranker(NeuralSummarizer):
    def __init__(self, 
                 setting='summary',    
                 num_sentences=0,
                 reverse=False,
                 output_list=False,
                 batch_size=4, 
                 device=None, 
                 enc_max_length=180, 
                 body_attr='text', 
                 query_attr='query',
                 out_attr='summary', 
                 verbose=False) -> None:
        super().__init__(None, None, batch_size, device, enc_max_length, body_attr, out_attr, verbose)
        
        self.num_sentences = num_sentences
        self.query_attr = query_attr
        self.reverse = 1 if reverse else -1
        if setting != 'scores' and output_list: setting = 'sentences'
        outputs = {
            'summary' : self.summary,
            'sentences' : self.list_summary,
            'scores' : self.scorer
        }
        self.output = outputs[setting]
        self.model = MonoT5Ranker()
    
    def _get_body(self,document):
        body = getattr(document, self.body_attr)
        sentences = split_into_sentences(body)
        if len(sentences) <= 1: return [body]
        return sentences
    
    def _construct_frame(self, query, sentences):
        idx = np.arange(len(sentences)).tolist()
        frame = pd.DataFrame({'docno' : idx, self.body_attr:sentences})
        frame['qid'] = 0
        frame[self.query_attr] = query
        return frame

    def _summary(self, sentences : List[str], scores : List[float]) -> str:
        idx = list(np.argsort(scores)[::self.reverse])
        if self.num_sentences != 0: return ' '.join([sentences[x] for x in idx][:self.num_sentences])
        return ' '.join([sentences[x] for x in idx])

    def _list_summary(self, sentences : List[str], scores : List[float]) -> str:
        idx = list(np.argsort(scores)[::self.reverse])
        if self.num_sentences != 0: return [sentences[x] for x in idx][:self.num_sentences]
        return [sentences[x] for x in idx][::self.reverse]

    def _scorer(self, sentences : List[str], scores : List[float]) -> List[float]:
        return scores
    
    def _summarize(self, text):
        sentences = self._get_body(text)
        if len(sentences) == 1: return self.output(sentences, [0])

        inp = self._construct_frame(getattr(text, self.query_attr), sentences)
        scores = self.model(inp)['score'].tolist()

        return self.output(sentences, scores)
    
    def transform(self, inp):
        assert self.query_attr in inp.columns and self.body_attr in inp.columns
        inp[self.out_attr] = inp.apply(lambda x : self._summarize(x), axis=1)
        return inp 
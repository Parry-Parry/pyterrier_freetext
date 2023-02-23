from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from . import split_into_sentences
from .. import NeuralSummarizer

class T5Ranker(NeuralSummarizer):
    def __init__(self, 
                 query_attr='query',
                 **kwargs) -> None:
        super().__init__(**kwargs)
        from pyterrier_t5 import MonoT5ReRanker

        self.query_attr = query_attr
        self.model = MonoT5ReRanker(batch_size=self.batch_size, text_field=self.body_attr, verbose=self.verbose)
    
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
    
class SentenceRanker(NeuralSummarizer):
    def __init__(self, 
                 model_name,
                 query_attr = 'query',
                 metric='cosine',    
                 batch_size=None, 
                 enc_max_length=180,
                 device=None,
                 **kwargs) -> None:
        super().__init__(model_name, None, batch_size, enc_max_length, device, **kwargs)
        from sentence_transformers import SentenceTransformer
        self.query_attr = query_attr
        self.metric = metric
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def _get_body(self,document):
        body = getattr(document, self.body_attr)
        sentences = split_into_sentences(body)
        if len(sentences) <= 1: return [body]
        return sentences
    
    def _summarize(self, text):
        sentences = self._get_body(text)
        if len(sentences) == 1: return self.output(sentences, [0])

        query_embedding = self.model.encode([getattr(text, self.query_attr)], convert_to_numpy=True)[0]
        sentence_embeddings = self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=self.verbose)
        scores = pairwise_distances(query_embedding.reshape(1, -1), np.stack(sentence_embeddings, axis=0), metric=self.metric)[0]
        return self.output(sentences, scores.tolist())
    
    def transform(self, inp):
        inp = inp.copy()
        assert self.query_attr in inp.columns and self.body_attr in inp.columns
        inp[self.out_attr] = inp.apply(lambda x : self._summarize(x), axis=1)
        return inp 
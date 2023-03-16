import pyterrier as pt
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from abc import abstractmethod
from more_itertools import chunked

class Summarizer(pt.Transformer):
    def __init__(self, 
                 body_attr='text', 
                 out_attr='summary', 
                 mode='summary', 
                 output_list=False, 
                 num_sentences=0, 
                 reverse=False, 
                 verbose=False) -> None:
        self.body_attr = body_attr
        self.out_attr = out_attr
        self.num_sentences = num_sentences
        self.reverse = 1 if reverse else -1 
        self.verbose = verbose

        if mode is None:
            self.output = None
        else:
            if mode != 'scores' and output_list: mode = 'sentences'
            outputs = {
                'summary' : self._summary,
                'sentences' : self._list_summary,
                'scores' : self._scorer,
                'ranks' : self._ranks
            }
            self.output = outputs[mode]
        
    def _summary(self, sentences : List[str], scores : np.ndarray) -> str:
        idx = np.argsort(scores).tolist()[::self.reverse]
        if self.num_sentences != 0: return ' '.join([sentences[x] for x in idx][:self.num_sentences])
        return ' '.join([sentences[x] for x in idx])

    def _list_summary(self, sentences : List[str], scores : np.ndarray) -> str:
        idx = np.argsort(scores).tolist()[::self.reverse]
        if self.num_sentences != 0: return [sentences[x] for x in idx][:self.num_sentences]
        return [sentences[x] for x in idx][::self.reverse]

    def _scorer(self, sentences : List[str], scores : np.ndarray) -> List[float]:
        return scores.tolist()
    
    def _ranks(self, sentences : List[str], scores : np.ndarray) -> List[int]:
        return np.argsort(scores).tolist()[::self.reverse]  
        
    @abstractmethod
    def transform(self, inp):
        raise NotImplementedError

class LexicalSummarizer(Summarizer):
    def __init__(self, indexref=None, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.indexref = indexref
        self.index = None  

    def init_index(self, documents):
        from pyterrier import DFIndexer, IndexFactory, autoclass
        from pyterrier.index import IndexingType

        indexref = DFIndexer(None, type=IndexingType.MEMORY, verbose=self.verbose).index(documents[self.body_attr], documents["docno"])
        docno2docid = {docno:id for id, docno in enumerate(documents["docno"])} # Keeping this mystery line just in case
        index_docs = IndexFactory.of(indexref)
        docno2docid = {index_docs.getMetaIndex().getItem("docno", i) : i for i in range(index_docs.getCollectionStatistics().getNumberOfDocuments())}
        assert len(docno2docid) == index_docs.getCollectionStatistics().getNumberOfDocuments(), "docno2docid size (%d) doesnt match index (%d)" % (len(docno2docid), index_docs.getCollectionStatistics().getNumberOfDocuments())
        if self.indexref is None:
            self.index = index_docs
        else:
            index_background = IndexFactory.of(self.indexref)
            self.index = autoclass("org.terrier.python.IndexWithBackground")(index_docs, index_background)

class NeuralSummarizer(Summarizer):
    def __init__(self, 
                 model_name=None, 
                 tokenizer_name=None, 
                 batch_size=32, 
                 enc_max_length=180, 
                 device=None, 
                 **kwargs) -> None:
        Summarizer.__init__(self, **kwargs)
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.enc_max_length = enc_max_length
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)        

class AbstractiveSummarizer(NeuralSummarizer):
    def __init__(self, 
                 model_name, 
                 inplace=True,
                 **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        
        self.inplace = inplace

        tokenizer_name = tokenizer_name if self.tokenizer_name else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
    
    def _set_mode(self, inp):
        if self.inplace : inp 
        else: return inp[["docno", self.body_attr]].drop_duplicates(subset="docno")
        
    def _summarize(self, text):
        results = []
        with torch.no_grad():
            for chunk in chunked(text.tolist(), self.batch_size):
                inps = self.tokenizer(chunk, padding="max_length", truncation=True, max_length=self.enc_max_length, return_tensors="pt")
                inps = inps.input_ids.to(self.device)
                attention_mask = inps.attention_mask.to(self.device)
                out = self.model.generate(inps, attention_mask=attention_mask).cpu()
                results.extend(self.tokenizer.batch_decode(out, skip_special_tokens=True))
        return results

    def transform(self, inp):
        assert "docno" in inp.columns and self.body_attr in inp.columns, "Malformed Frame, Expecting Documents"
        out = self._set_mode(inp)
        out[self.out_attr] = self._summarize(out[self.body_attr])
        return out 
    
from ranker import MonoT5SentenceRanker, SentenceRanker
from ranker.lexrank import LexRanker
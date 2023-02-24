from typing import List

import numpy as np
import pyterrier as pt
import torch
from abc import abstractmethod

class Summarizer(pt.Transformer):
    def __init__(self, 
                 body_attr, 
                 out_attr, 
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
    def __init__(self, indexref, **kwargs) -> None:
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
                 model_name, 
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

def sentencesnippets(
        text_scorer_pipe : pt.Transformer, 
        text_attr : str = "text", 
        summary_attr : str = "summary") -> pt.Transformer:
    """
    Applies query-biased summarisation (snippet), by applying the specified text scoring pipeline.

    Parameters:
        text_scorer_pipe(Transformer): the pipeline for scoring passages in response to the query. Normally this applies passaging.
        text_attr(str): what is the name of the attribute that contains the text of the document
        summary_attr(str): what is the name of the attribute that should contain the query-biased summary for that document

    Example::

        # retrieve documents with text
        br = pt.BatchRetrieve(index, metadata=['docno', 'text'])

        # use Tf as a passage scorer on sliding window passages 
        psg_scorer = ( 
            pt.text.sliding(text_attr='text', length=15, prepend_attr=None) 
            >> pt.text.scorer(body_attr="text", wmodel='Tf', takes='docs')
        )
        
        # use psg_scorer for performing query-biased summarisation on docs retrieved by br 
        retr_pipe = br >> pt.text.snippets(psg_scorer)

    """
    tsp = (
        pt.apply.rename({'qid' : 'oldqid'}) 
        >> pt.apply.qid(lambda row: row['oldqid'] + '-' + row['docno']) 
        >> text_scorer_pipe
        >> pt.apply.qid(drop=True)
        >> pt.apply.rename({'oldqid' : 'qid'})
    )

    def _qbsjoin(docres):
        import pandas as pd
        if len(docres) == 0:
            docres[summary_attr] = pd.Series(dtype='str')
            return docres     

        psgres = tsp(docres)
        if len(psgres) == 0:
            print('no passages found in %d documents for query %s' % (len(docres), docres.iloc[0].query))
            docres = docres.copy()
            docres[summary_attr]  = ""
            return docres

        psgres[["olddocno", "pid"]] = psgres.docno.str.split("%p", expand=True)

        newdf = psgres.groupby(['qid', 'olddocno'])[text_attr].reset_index().rename(columns={text_attr : summary_attr, 'olddocno' : 'docno'})
        
        return docres.merge(newdf, on=['qid', 'docno'], how='left')
    return pt.apply.generic(_qbsjoin)   
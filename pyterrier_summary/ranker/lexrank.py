import pyterrier as pt
import logging
from collections import Counter, defaultdict
import math
from typing import List, NamedTuple, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from . import split_into_sentences
from .. import LexicalSummarizer

class LexRanker(LexicalSummarizer):
    def __init__(self, 
                 documents=None,
                 threshold=0., 
                 norm=True,
                 tokeniser='english', 
                 stopwords=True,
                 stemmer='PorterStemmer', 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        """LexRank Transformer
        ----------------------
        Settings:
            summary -- Returns specified number of sentences ranked by salience in ascending or descending order joined as a string
            scores -- Returns salience scores in sentence order as a list
        ----------------------
        Kwargs:
            documents -- Corpus to initialise index
            background_index -- Terrier indexref to intialise index
            threshold -- Threshold for quantisation ~ [0, 1], leave as 0. for no quantisation
            num_sentences -- How many sentences to use in summary, leave as 0 to just rank sentences
            reverse -- If True, take least salient sentences
            norm -- If True, normalise salience scores
            output_list -- Flag for output of sentences in list
            tokeniser -- str name of Terrier tokeniser
            stopwords -- Flag to include stopword removal in pipeline
            stemmer -- name or Java API reference of Terrier term stemmer 
            body_attr -- Attribute from which to retrieve text for sentence ranking
            out_attr -- Column name for summary output
            verbose -- verbose output

        Markov Stationary Distribution Computation based on https://github.com/crabcamp/lexrank/
        """

        self.lexicon = None
        self.N = None

        self.threshold = threshold
        self.norm = norm

        self.tokeniser = pt.rewrite.tokenise(tokeniser=tokeniser)

        if stopwords: self.stopwords = pt.autoclass("org.terrier.terms.Stopwords")(None).isStopword
        else: self.stopwords = lambda x : False

        if stemmer is not None:
            stem_name = f"org.terrier.terms.{stemmer}" if '.' not in stemmer else stemmer
            self.stemmer = pt.autoclass(stem_name)().stem
        else:
            self.stemmer = lambda x : x

        if documents: self.init_index(documents)

    def _text_pipeline(self, text):
        """Tokenise sentences, stem and remove stopwords"""
        tokenised = [sentence.split() for sentence in self.tokeniser(pd.DataFrame({'query':text}))['query'].tolist()]

        stemmed = [] 
        for sentence in tokenised:
            stemmed.append([self.stemmer(term) for term in sentence if not self.stopwords(term)])
        return stemmed
    
    def _tf(self, document : NamedTuple) -> Tuple[dict, list]:
        """Split, tokenize and stem sentences then compute term frequencies"""
        body = getattr(document, self.body_attr)
        sentences = split_into_sentences(body) 
        if len(sentences) == 0: sentences = [body]
        stemmed = self._text_pipeline(sentences)
        tf = {i : Counter(sentence) for i, sentence in enumerate(stemmed)}
        return tf, sentences
    
    def _idf(self, token):
        try:
            df = self.lexicon[token].getDocumentFrequency()
        except KeyError:
            logging.warning(f"{token} not found in Lexicon, returning 0")
            return 0.
        return math.log(self.N / df)
    
    def _idf_cosine(self, i : dict, j : dict) -> float:
        """Computed IDF modified cosine similarity between two sentences i and j"""
        if i==j: return 1. 
        tokens_i, tokens_j = set(i.keys()), set(j.keys())

        accum = 0
        idf_scores = defaultdict(float)
        for token in tokens_i & tokens_j:
            idf_score = self._idf(token)
            idf_scores[token] = idf_score
            accum += i[token] * j[token] * idf_score ** 2
        
        if math.isclose(accum, 0.): return 0.

        mag_i, mag_j = 0, 0

        for token in tokens_i:
            idf_score = idf_scores[token]
            if idf_score == 0.: self._idf(token)
            tfidf = i[token] * idf_score
            mag_i += tfidf ** 2
        
        for token in tokens_j:
            idf_score = idf_scores[token]
            if idf_score == 0.: idf_score = self._idf(token)
            tfidf = j[token] * idf_score
            mag_j += tfidf ** 2
        
        return accum / math.sqrt(mag_i * mag_j)
    
    def _markov_matrix(self, matrix : np.ndarray) -> np.ndarray:
        """Normalise to create probabilities"""
        if matrix.shape[0] != matrix.shape[1]: raise ValueError('matrix should be square')
        row_sum = matrix.sum(axis=1, keepdims=True)

        return matrix / row_sum

    def _quantized_markov_matrix(self, matrix : np.ndarray) -> np.ndarray:
        """Quantize similarity matrix ~ [0, 1]"""
        _matrix = np.zeros(matrix.shape)
        idx = np.where(matrix >= self.threshold)
        _matrix[idx] = 1

        return self._markov_matrix(_matrix)
    
    def _connected_nodes(self, matrix : np.ndarray) -> List:
        """Get adjacency matrix"""
        _, labels = connected_components(matrix)
        return [np.where(labels == tag)[0] for tag in np.unique(labels)]

    def _power_method(self, matrix : np.ndarray) -> np.ndarray:
        """Power iteration until convergence"""
        eigenvector = np.ones(matrix.shape[0])
        if eigenvector.shape[0] == 1: return eigenvector

        transition = matrix.T
        while True:
            _next = transition @ eigenvector

            if np.allclose(_next, eigenvector):
                if self.verbose: logging.debug('Converged')
                return _next

            eigenvector = _next
            transition = transition @ transition

    def _stationary_distribution(self, matrix : np.ndarray) -> np.ndarray:
        "Get LexRank Score via eigenvector of transformed similarity matrix"
        distribution = np.zeros(matrix.shape[0])
        grouped_indices = self._connected_nodes(matrix)

        for group in grouped_indices:
            t_matrix = matrix[np.ix_(group, group)]
            eigenvector = self._power_method(t_matrix)
            distribution[group] = eigenvector

        if self.norm:
            distribution /= matrix.shape[0]

        return distribution

    def lexrank(self, doc) -> Union[str, List[float], List[str]]:
        assert self.index is not None, 'Initialize Index'
        logging.debug(f'Computing LexRank for Doc:{doc.docno}')
        # Get sentence level term frequencies
        tf_scores, sentences = self._tf(doc) 
        if len(sentences) == 1: return self.output(sentences, [0])
        
        # Construct similarity matrix
        dim = len(tf_scores)
        sim_matrix = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                sim = self._idf_cosine(tf_scores[i], tf_scores[j])
                if sim:
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        
        # Compute Stationary Distribution 
        transition = self._quantized_markov_matrix(sim_matrix) if self.threshold !=0. else self._markov_matrix(sim_matrix)
        scores = self._stationary_distribution(transition)

        return self.output(sentences, scores)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert "docno" in inp.columns and self.body_attr in inp.columns, "Malformed Frame, Expecting Documents"
        inp = inp.copy()
        if self.index is None:
             logging.warning('Index not initialized, creating from inputs and a reference if it exists')   
             self.init_index(inp)
            
        self.lexicon = self.index.getLexicon()
        self.N = self.index.getCollectionStatistics().getNumberOfDocuments()

        inp[self.out_attr] = inp.apply(lambda x : self._lexrank(x), axis=1)
        return inp
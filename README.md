# PyTerrier Summary

This is a [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for summarisation.

## Usage

This package includes Terrier compatible components for ranking sentences in text or directly summarising text using seq2seq models. Models exist both for contextualized ranking with respect to a query text or unsupervised methods that determine sentence centrality with optional background corpus statistics from Terrier.

## Implementation Details

Neural summarisers use Hugging Face transformer models aswell as some pyterrier plugins. If using the T5ranker you will need to install [Pyterrier T5](https://github.com/terrierteam/pyterrier_t5/).
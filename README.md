# PyTerrier FreeText

This is a [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for summarisation.

## Usage

This package allows PyTerrier users to more easily include free text transformations in their retrieval pipelines. Examples of this include generative models using custom prompts aswell as abstractive and lexical summarization with sentence ranking. Models exist both for contextualized ranking with respect to a query text or unsupervised methods that determine sentence centrality with optional background corpus statistics from Terrier.

## Implementation Details

Neural pipelines use Hugging Face transformer models aswell as some pyterrier plugins.

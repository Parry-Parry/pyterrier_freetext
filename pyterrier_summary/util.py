import pyterrier as pt
import re
from typing import List 

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

def sentence_snippets(
        text_scorer_pipe : pt.Transformer, 
        text_attr : str = "text", 
        summary_attr : str = "summary", 
        num_psgs : int = None, 
        joinstr : str = None) -> pt.Transformer:
    """
    Applies query-biased summarisation (snippet), by applying the specified text scoring pipeline.

    Parameters:
        text_scorer_pipe(Transformer): the pipeline for scoring passages in response to the query. Normally this applies passaging.
        text_attr(str): what is the name of the attribute that contains the text of the document
        summary_attr(str): what is the name of the attribute that should contain the query-biased summary for that document
        num_psgs(int): how many passages to select for the summary of each document
        joinstr(str): how to join passages for a given document together

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
    import pyterrier as pt

    scorer = text_scorer_pipe % num_psgs if num_psgs else text_scorer_pipe
    tsp = (
        pt.apply.rename({'qid' : 'oldqid'}) 
        >> pt.apply.qid(lambda row: row['oldqid'] + '-' + row['docno']) 
        >> scorer
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

        new_text = psgres.groupby(['qid', 'olddocno'])[text_attr]
        if joinstr: new_text = new_text.agg(joinstr.join)
        newdf = new_text.reset_index().rename(columns={text_attr : summary_attr, 'olddocno' : 'docno'})
        
        return docres.merge(newdf, on=['qid', 'docno'], how='left')
    return pt.apply.generic(_qbsjoin)   

def split_into_sentences(text : str) -> List[str]:
    """Sentence Regex from: https://stackoverflow.com/a/31505798"""
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def splitter(df, body_attr='text'):
    df['sentences'] = df[body_attr].apply(lambda x : split_into_sentences(x), axis=1)
    return df

def compose_pipe(data_pipe, summarizer):
    summary = sentence_snippets(summarizer, text_attr=summarizer.body_attr, summary_attr='summary') 
    return data_pipe >> summary 
from pyterrier_freetext.summary.ranker import SentenceRanker
import ir_datasets
import pandas as pd
import fire

def build_data(path):
  result = []
  dataset = ir_datasets.load(path)
  docs = dataset.docs_store()
  queries = {q.query_id: q.text for q in dataset.queries_iter()}
  for qrel in ir_datasets.load(path).scoreddocs_iter():
    if qrel.query_id in queries:
      result.append([qrel.query_id, queries[qrel.query_id], qrel.doc_id, docs.get(qrel.doc_id).text])
  return pd.DataFrame(result, columns=['qid', 'query', 'docno', 'text'])

def main(dataset : str, summary_model : str, num_docs : int, mode='summary'):
    data = build_data(dataset).iloc[:num_docs]

    model = SentenceRanker(summary_model, mode=mode, num_sentences=1, out_attr='sentence')
    out = model.transform(data)
    print(out.text, out.sentence)

if __name__=='__main__':
    fire.Fire(main)
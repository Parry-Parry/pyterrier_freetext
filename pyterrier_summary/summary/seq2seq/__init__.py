from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from more_itertools import chunked

from .. import NeuralSummarizer

class Seq2SeqSummarizer(NeuralSummarizer):
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
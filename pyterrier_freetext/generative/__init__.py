import pyterrier as pt
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json

class Prompt:
    # Prompt constructor from LightChain
    def __init__(self, prompt, params=None):
        self.prompt = prompt
        self.params = params

        if params:
            for param in params: assert f'{{{param}}}' in prompt, f'Param {param} not found in prompt {prompt}'
        
    def __str__(self):
        return self.prompt

    def __repr__(self):
        return f'Prompt(prompt={self.prompt}, params={self.params})'
    
    @staticmethod
    def fromjson(json_str):
        return json.loads(json_str, object_hook=lambda x: Prompt(**x))
                          
    def tojson(self):
        return json.dumps(self, default=lambda x: x.__dict__, 
            sort_keys=True, indent=4)
    
    def construct(self, **kwargs):
        for key in kwargs: assert key in self.params, f'Param {key} not found in params {self.params}'
        return self.prompt.format(**kwargs)
    
    def batch_construct(self, params):
        return [self.construct(**param) for param in params]
    
class GenericGenerativeTransformer(pt.Transformer):
    def __init__(self, 
                 prompt : Prompt, 
                 out_attr : str ='text', 
                 post_process : callable = lambda x : x,
                 **kwargs) -> None:
        super().__init__()
        self.prompt = prompt 
        self.out_attr = out_attr
        self.post_process = post_process
        self.generate = None
        self.model = None 
        self.tokenizer = None

        self.batch_size = kwargs.pop('batch_size', 1)

        self.generate_kwargs = {
            "max_new_tokens": kwargs.pop('max_tok', 128),
            "min_new_tokens": kwargs.pop('max_tok', 0),
            "temperature": kwargs.pop('temperature', 0.8),
            "do_sample": kwargs.pop('constrastive_search', False),
            "top_p" : kwargs.pop('top_p', 0.95), 
            "top_k": kwargs.pop('top_k', 5),
            "penalty_alpha": kwargs.pop('penalty_alpha', 0.6),
            "repetition_penalty": kwargs.pop('repetition_penalty', 1.0),
            "length_penalty" : kwargs.pop('length_penalty', 1.0)
        }

    def batch_generate(self, df):
        sub = df[self.prompt.get_params()].todict('records')
        text = self.prompt.batch_prompt(sub)

        # Change to custom batching

        pipe = pipeline(model=self.model, tokenizer=self.tokenizer, batch_size=self.batch_size)
        return list(map(self.post_process, pipe(text)))

    def fit(self, train_data) -> None:
        raise NotImplementedError

    def transform(self, input):
        assert self.generate is not None, "Must instantiate a model!"
        output = input.copy()
        if self.batch_size > 1: output[self.out_attr] = self.batch_generate(output)
        else: output[self.out_attr] = output.apply(lambda x : self.generate(x), axis=1)
        return output

class FlexibleLMTransformer(GenericGenerativeTransformer):
    def __init__(self, 
                 prompt : Prompt, 
                 model, 
                 tokenizer,
                 out_attr : str ='text', 
                 post_process : callable = lambda x : x,
                 generate_kwargs : dict = None,
                 **kwargs) -> None:
        super().__init__(prompt=prompt, out_attr=out_attr, post_process=post_process, **kwargs)
        self.prompt = prompt 
        self.out_attr = out_attr 
        self.post_process = post_process
        
        self.tokenizer = tokenizer
        self.model = model

        if generate_kwargs:
            self.generate_kwargs = generate_kwargs

    def generate(self, frame) -> str:
        sub = frame[self.prompt.get_params()].todict()
        if self.batch_size > 1: text = self.prompt.batch_prompt(sub)
        else: text = self.prompt.create_prompt(**sub)
        with torch.no_grad():
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            input_ids = input_ids.to(0)
        
            generated_ids = self.model(
                input_ids,
                **self.generate_kwargs
            )

        out = list(map(self.post_process, self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)))
        if self.batch_size > 1: return out
        return out[0]

class CausalLMTransformer(GenericGenerativeTransformer):
    def __init__(self,
                 prompt : Prompt, 
                 model_id : str, 
                 out_attr : str ='text', 
                 post_process : callable = lambda x : x,
                 **kwargs) -> None:
        super().__init__(prompt=prompt, out_attr=out_attr, post_process=post_process, **kwargs)
        self.prompt = prompt 
        self.out_attr = out_attr 
        self.post_process = post_process
        
        do_int8 = kwargs.pop('do_int8', False)
        low_cpu_mem_usage = kwargs.pop('low_cpu_mem_usage', False)
        device_map = kwargs.pop('device_map', 'auto')

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.int8 if do_int8 else torch.float16,
            low_cpu_mem_usage=True if low_cpu_mem_usage else None,
            load_in_8bit=do_int8
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast="/opt" not in model_id)

    def generate(self, frame) -> str:
        sub = frame[self.prompt.get_params()].todict()
        if self.batch_size > 1: text = self.prompt.batch_prompt(sub)
        else: text = self.prompt.create_prompt(**sub)
        with torch.no_grad():
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            input_ids = input_ids.to(0)
        
            generated_ids = self.model.generate(
                input_ids,
                **self.generate_kwargs
            )
        return list(map(self.post_process, self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)))

def causal_transform(prompt, model_id, **kwargs):
    transform = CausalLMTransformer(prompt, model_id, **kwargs)
    return pt.generic_apply(lambda x : transform(x))
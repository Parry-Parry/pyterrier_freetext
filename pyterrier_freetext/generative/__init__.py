import pyterrier as pt
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PromptConstructor:
    def __init__(self, prompt, params) -> None:
        for param in params: assert f"[{param}]" in prompt, "Parameter not found in prompt"
        self.prompt = prompt
        self.params = params 
    
    def get_params(self):
        return self.params
        
    def create_prompt(self, **kwargs):
        tmp_prompt = self.prompt
        if self.params:
            for param in self.params: tmp_prompt = re.sub(f'[{param}]', tmp_prompt, kwargs.pop(param, "")) 
        return tmp_prompt
    
class GenericGenerativeTransformer(pt.Transformer):
    def __init__(self, 
                 prompt : PromptConstructor, 
                 out_attr : str ='text', 
                 post_process : callable = lambda x : x,
                 **kwargs) -> None:
        super().__init__()
        self.prompt = prompt 
        self.out_attr = out_attr
        self.post_process = post_process
        self.generate = None

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

    def transform(self, input):
        assert self.generate is not None, "Must instantiate a model!"
        output = input.copy()
        output[self.out_attr] = output.apply(lambda x : self.generate(x), axis=1)

class FlexibleLMTransformer(GenericGenerativeTransformer):
    def __init__(self, 
                 prompt : PromptConstructor, 
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
    
    def fit(self, train_data) -> None:
        raise NotImplementedError

    def generate(self, frame) -> str:
        sub = frame[self.prompt.get_params()].todict()
        text = self.prompt.create_prompt(**sub)
        with torch.no_grad():
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            input_ids = input_ids.to(0)
        
            generated_ids = self.model(
                input_ids,
                **self.generate_kwargs
            )
            return self.post_process(self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0])

class CausalLMTransformer(GenericGenerativeTransformer):
    def __init__(self,
                 prompt : PromptConstructor, 
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
    
    def fit(self, train_data) -> None:
        raise NotImplementedError

    def generate(self, frame) -> str:
        sub = frame[self.prompt.get_params()].todict()
        text = self.prompt.create_prompt(**sub)
        with torch.no_grad():
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            input_ids = input_ids.to(0)
        
            generated_ids = self.model.generate(
                input_ids,
                **self.generate_kwargs
            )
            return self.post_process(self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0])

def causal_transform(prompt, model_id, **kwargs):
    transform = CausalLMTransformer(prompt, model_id, **kwargs)
    return pt.generic_apply(lambda x : transform(x))
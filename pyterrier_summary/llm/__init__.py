import re
import pyterrier as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

class prompt_constructor:
    def __init__(self, prompt, params) -> None:
        for param in params: assert f"[{param}]" in prompt, "Parameter not found in prompt"
        self.prompt = prompt
        self.params = params 
    
    def get_params(self):
        return self.params
        
    def create_prompt(self, **kwargs):
        tmp_prompt = self.prompt
        if self.names:
            params = {param : kwargs.pop(param, "") for param in self.params}
            for param in self.names: tmp_prompt = re.sub(f'[{param}]', tmp_prompt, params[param]) 
        return tmp_prompt

class GenerativeTransformer(pt.Transformer):
    def __init__(self, 
                 prompt, 
                 model_id, 
                 out_attr='text', 
                 **kwargs) -> None:
        super().__init__()
        self.prompt = prompt 
        self.out_attr = out_attr 
        self.batch_size = kwargs.pop('batch_size', 1)
        
        self.tokenizer = AutoTokenizer(model_id)
        
        do_int8 = kwargs.pop('do_int8', False)
        low_cpu_mem_usage = kwargs.pop('low_cpu_mem_usage', False)
        device_map = kwargs.pop('device_map', 'auto')
        # Set up accelerate
        # Is the model sharded?
        # Other generation kwargs
    
    def fit(self, train) -> None:
        pass

    def generate(self, frame) -> str:
        sub = frame[self.prompt.get_params()].todict()
        text = self.prompt(**sub)
        input_ids = self.tokenizer(text).input_ids.to(0)

    def transform(self, input):
        output = input.copy()
        output[self.out_attr] = output.apply(lambda x : self.generate(x))

def generic_transform(prompt, model_id, **kwargs):
    transform = GenerativeTransformer(prompt, model_id, **kwargs)
    return pt.generic_apply(lambda x : transform(x))
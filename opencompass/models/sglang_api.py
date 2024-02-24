from .base_api import BaseAPIModel
from typing import Dict, List, Optional, Union
from sglang import function, set_default_backend, system, user, assistant, gen, RuntimeEndpoint
from opencompass.utils.prompt import PromptList
import tiktoken
class SglangAPI(BaseAPIModel):
    def __init__(self,
                 path: str='default',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 retry: int = 2,
                 meta_template: dict = None,
                 temperature: float = 0,
                 url: str = 'https://localhost:30000',
                 ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)
        self.url = url
        self.temperature = temperature
        self.tiktoken = tiktoken
        set_default_backend(RuntimeEndpoint(url))
    def generate(
        self,
        inputs:List[Union[str, PromptList]],
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        if self.temperature is not None:
            temperature = self.temperature
        states = self._generate.run_batch([{'input':input} for input in inputs],temperature=temperature,max_new_tokens=max_out_len)
        return [state['response'] for state in states]

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        enc = self.tiktoken.encoding_for_model('gpt-3.5-turbo')
        return len(enc.encode(prompt))

    @function
    def _generate(s,input):
        if isinstance(input, str):
            s += user(input)
            s += assistant(gen('response'))
        else:
            for item in input:
                if item['role'] == 'HUMAN':
                    s += user(item['prompt'])
                elif item['role'] == 'BOT':
                    s += assistant(item['prompt'])
                elif item['role'] == 'SYSTEM':
                    s += system(item['prompt'])
            s += assistant(gen('response'))
from typing import Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llm.basellm import BaseLLM
# from ctransformers import AutoModelForCausalLM




class LlamaChat(BaseLLM):
    """Wrapper around Llama 2 large language models."""

    def __init__(
        self,
        model_name_or_path: str = "TheBloke/Llama-2-7B-32K-Instruct-GPTQ",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> None:
        print('getting the model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             torch_dtype=torch.float16,
                                             device_map="cuda",
                                             revision="main")
        # self.model = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-7B-Chat-GGML', model_file = 'llama-2-7b-chat.ggmlv3.q4_K_S.bin' )
        print('Done!!!!!!')
        self.max_tokens = max_tokens
        self.temperature = temperature
        # self.device = "cpu"
        # self.model.to(self.device)
    def generate(self, messages: List[str]) -> str:
        try:
            result_string = ""
            for item in messages:
                result_string += f"{item['role']}: {item['content']}\n"
            print('before tokenizer ',messages)
            input_ids = self.tokenizer(result_string, return_tensors="pt").input_ids.to(self.model.device)
            print('before generating ')
            output = self.model.generate(
                input_ids,
                max_length=1000,  # Adjust max length as needed
                temperature=self.temperature,  # Use the temperature you desire
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print('finished generate')
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print('finished generate!!!!',generated_text)
            return generated_text
        except Exception as e:
            return str(f"Error: {e}")

    async def generateStreaming(
    self, messages: List[dict], onTokenCallback
) -> List[Any]:
        try:
            result_string = ""
            for item in messages:
                result_string += f"{item['role']}: {item['content']}\n"

            input_ids = self.tokenizer(result_string, return_tensors="pt").input_ids.to(self.model.device)
            print('stream start!!!!!!!')
            with torch.no_grad():
                output = self.model.generate(input_ids, temperature=self.temperature)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print('stream end!!!!!!!',generated_text)
            tokens = generated_text.split()
            print('stream end2!!!!!!!',tokens)
            for token in tokens:
                await onTokenCallback(token)
            print('stream end3!!!!!!!',token)
            print('stream done!!!!!!!')
            return tokens
        except Exception as e:
            print(str(f"Error: {e}"))
            return str(f"Error: {e}")

    async def num_tokens_from_string(self, string: str) -> int:
        encoding = self.tokenizer(string, return_tensors="pt")
        num_tokens = encoding.input_ids.size(1)
        return num_tokens

    async def max_allowed_token_length(self) -> int:
        # You can set the max token length based on your specific requirements
        return 4096  # Set an appropriate value here

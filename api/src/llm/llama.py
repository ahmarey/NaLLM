from typing import Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llm.basellm import BaseLLM



class LlamaChat(BaseLLM):
    """Wrapper around Llama 2 large language models."""

    def __init__(
        self,
        model_name_or_path: str = "TheBloke/Llama-2-7B-32K-Instruct-GPTQ",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             torch_dtype=torch.float16,
                                             device_map="cuda",
                                             revision="main")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = "cpu"
        self.model.to(self.device)
    def generate(self, messages: List[str]) -> str:
        try:
            inputs = self.tokenizer(
                messages,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_tokens,
                truncation=True,
            )
            with torch.no_grad():
                output = self.model.generate(**inputs, temperature=self.temperature)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            return str(f"Error: {e}")

    async def generateStreaming(
        self, messages: List[str], onTokenCallback
    ) -> List[Any]:
        try:
            inputs = self.tokenizer(
                messages,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_tokens,
                truncation=True,
            )
            with torch.no_grad():
                output = self.model.generate(**inputs, temperature=self.temperature)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            tokens = generated_text.split()
            for token in tokens:
                await onTokenCallback(token)
            return tokens
        except Exception as e:
            return str(f"Error: {e}")

    async def num_tokens_from_string(self, string: str) -> int:
        encoding = self.tokenizer(string, return_tensors="pt")
        num_tokens = encoding.input_ids.size(1)
        return num_tokens

    async def max_allowed_token_length(self) -> int:
        # You can set the max token length based on your specific requirements
        return 4096  # Set an appropriate value here

import textwrap
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

MODEL_NAME = "TheBloke/stable-vicuna-13B-HF"

tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)

model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map="auto",
    offload_folder="./cache",
)

def format_prompt(prompt: str) -> str:
    text = f"""
### Human: {prompt}
### Assistant:
    """
    return text.strip()


print(format_prompt("What is your opinion on ChatGPT? Reply in 1 sentence."))
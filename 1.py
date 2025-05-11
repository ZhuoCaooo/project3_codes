from llama_cpp import Llama
import torch
from transformers import AutoTokenizer

# Load the GGUF model
model_path = "gemma-2b-instruct-ft-awesome-chatgpt-prompts-Q4_K.gguf"  # Update with your actual filename
model = Llama(model_path=model_path, n_ctx=2048)  # Context window size

# Load tokenizer separately
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
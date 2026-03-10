from typing import List, Tuple, Dict
from tiktoken import get_encoding # this is the library used by OpenAI for tokenization and is compatible with their models
from transformer.tokenization_utils_base import PreTrainedTokenizerBase # we will use this as a base class for our custom tokenizer wrapper this libary is used by HuggingFace transformers and is compatible with many models including OpenAI's

#lets create a custom tokenizer wrapper that uses tiktoken under the hood but provides a simple interface for our use case for HybridChunker and DocumentConverter. This wrapper will handle tokenization and detokenization, as well as counting tokens in a given text.
class OpenAITokenizerWrapper(PreTrainedTokenizerBase):
  def __init__(self, model_name: str = "text-embedding-3-large"):
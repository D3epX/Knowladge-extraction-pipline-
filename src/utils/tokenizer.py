from typing import List, Tuple, Dict
from tiktoken import get_encoding # this is the library used by OpenAI for tokenization and is compatible with their models
from transformer.tokenization_utils_base import PreTrainedTokenizerBase # we will use this as a base class for our custom tokenizer wrapper this libary is used by HuggingFace transformers and is compatible with many models including OpenAI's

#lets create a custom tokenizer wrapper that uses tiktoken under the hood but provides a simple interface for our use case for HybridChunker and DocumentConverter. This wrapper will handle tokenization and detokenization, as well as counting tokens in a given text.
#   """Minimal wrapper for OpenAI's tokenizer."""
#OpenAITokenizerWrapper
#        ↓
#behaves like HuggingFace tokenizer
'''| Parameter    | Meaning                   |
   | ------------ | ------------------------- |
   | `model_name` | OpenAI tokenizer encoding |--> default encoding is cl100k_base
   | `max_length` | maximum token length      |--> 8191

   **kwargs --> "accept any number of keyword arguments and store them in a dictionary (kwargs = dictionary of named parameters)"
      example: def func(**kwargs):
    print(kwargs)/ 
    func(x=1, y=2) --> {'x':1, 'y':2} '''
class OpenAITokenizerWrapper(PreTrainedTokenizerBase):
  def __init__(self, model_name: str = "cl100k_base", max_length: int = 8191, **kwargs):
       super.__init__(model_max_length = max_length, **kwargs) # call the parent class constructor
       self.tokenizer = get_encoding(model_name) # initialize the tiktoken tokenizer with the specified encoding ,getencoding is a function from tiktoken that returns a tokenizer object based on the specified encoding name
       self.vocab_size = self.tokenizer.max_token_value # get the maximum token value from the tokenizer which represents the size of the vocabulary
  def tokenize(self, text: str, **kwargs) -> List[int]: #main function used to tokenize text, it takes a string input and returns a list of token used by hybridchunker and documentconverter to convert text into tokens for processing by the OpenAI models
       return[str(t)for t in self.tokenizer.encode(text)]# use the tiktoken tokenizer to encode the text into token ids
  def _tokenize(self, text: str) -> List[str]:
      return self.tokenize(text) # this is a required method for compatibility with HuggingFace tokenizers, it simply calls the tokenize method we defined above simply calls main tokenizer .
  def _convert_token_to_id(self, token:str) -> int:
      return int(token) # convert the token string back to an integer id
  def _convert_id_to_token(self,index :int) -> str:
      return str(index) # convert the integer id back to a token string
  def get_vocab(self) -> Dict[str , int]:
      return dict(enumerate(range(self.vocab_size))) # return a dictionary mapping token strings to their corresponding integer ids based on the vocabulary size of the tokenizer , enumerate is a function that adds a counter to an iterable and returns it as an enumerate object, which can be converted to a list of tuples or a dictionary
  @property # this is a property decorator that allows us to access the vocab_size as an attribute rather than a method
  def vocab_size(self) -> int:
      return self._vocab_size # return the vocabulary size of the tokenizer
  def save_vocabulary(self, *args) -> Tuple[str]:
      return() # this is a required method for compatibility with HuggingFace tokenizers, it can be used to save the tokenizer's vocabulary to a file, but since tiktoken does not require this we can simply return an empty tuple and *args allows us to accept any number of arguments without causing an error
  @classmethod # this is a class method decorator that allows us to create an instance of the tokenizer from a pretrained model name, but since our tokenizer does not require this we can simply return an instance of the class 
  def from_pretrained(cls, *args, **kwargs):
      return cls() # return an instance of the OpenAITokenizerWrapper class, we can ignore the *args and **kwargs since our tokenizer does not require any additional parameters for initialization and cls is a reference to the class itself, allowing us to create an instance of the class without needing to know its name or how it is defined.
  
  '''so as recap from previous comments what we learned is that this OpenAITokenizerWrapper class is a custom tokenizer wrapper that uses the tiktoken library to provide tokenization ,
  and detokenization functionality for OpenAI models.
  It is designed to be compatible with HuggingFace tokenizers, allowing it to be easily integrated into our HybridChunker and DocumentConverter classes. 
  The wrapper provides methods for tokenizing text, converting tokens to ids and vice versa, and retrieving the vocabulary size, while also handling any additional parameters through the use of **kwargs.
  we have also implemented the required methods for compatibility with HuggingFace tokenizers, such as _tokenize, _convert_token_to_id, _convert_id_to_token, get_vocab, save_vocabulary, and from_pretrained, even though some of these methods may not be necessary for our specific use case with tiktoken. 
  overall this wrapper allows us to easily tokenize and detokenize text for use with OpenAI models while maintaining compatibility with the HuggingFace ecosystem. 
  and we have also set a default maximum token length of 8191, which is the context length for the text-embeddings-3-large model, ensuring that our tokenization process is optimized for the models we intend to use. 
  at last we added a property for vocab_size to allow easy access to the vocabulary size of the tokenizer, which can be useful for various applications such as determining the size of the input or output space for our models. 
  overall this implementation provides a robust and flexible tokenizer wrapper that can be easily integrated into our document processing pipeline while ensuring compatibility with both tiktoken and HuggingFace tokenizers, and class method from_pretrained allows us to create an instance of the tokenizer without needing to know the specific details of its implementation, making it easier to use and integrate into our codebase(matches the interface of HuggingFace tokenizers). '''
  
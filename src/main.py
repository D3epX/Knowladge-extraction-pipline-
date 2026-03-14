from utils.tokenizer import OpenAITokenizerWrapper
# Example usage of the OpenAITokenizerWrapper

def main():
    #create an instance of the tokenizer 
    tokenizer = OpenAITokenizerWrapper()

    #example of text to tokenize 
    text = "hello how are you doinn"
    tokens =tokenizer.tokenize(text)

    print("Original text:") 
    print(text)
    print("\nTokens:") 
    print(tokens)
    print("\nToken IDs:")
    tokens_ids = [ tokenizer._convert_token_to_id(t) for t in tokens]
    print(tokens_ids)
    print("\nConvert IDs back to tokens:")
    tokens_from_ids = [tokenizer._convert_id_to_token(i) for i in tokens_ids]
    print(tokens_from_ids)

    print("\nVocabulary size:")
    print(tokenizer.vocab_size)

if __name__ == "__main__":
       main()

'''This code demonstrates how to use the OpenAITokenizerWrapper class to tokenize
there is the output of the code when run:
Original text:
hello how are you doinn

Tokens:
['15339', '1268', '527', '499', '656', '6258']

Token IDs:
[15339, 1268, 527, 499, 656, 6258]

Convert IDs back to tokens:
['15339', '1268', '527', '499', '656', '6258']

Vocabulary size:
100276  

| Token | Token ID |
| ----- | -------- |
| hello | 15339    |
| how   | 1268     |
| are   | 527      |
| you   | 499      |
| do    | 656      |
| inn   | 6258     |

doinn → do + inn llm tokenizer breaks down the word "doinn" into two tokens "do" and "inn" because it is not a common word in the tokenizer's vocabulary, so it splits it into smaller subword units that are more likely to be recognized by the model. This is a common behavior of tokenizers used in language models, as they often need to handle a wide variety of words and subwords to effectively process natural language text.
TEXT
  ↓
tiktoken
  ↓
[15339, 1268, 527, 499]
  ↓
wrapper converts to
  ↓
["15339","1268","527","499"]
'''

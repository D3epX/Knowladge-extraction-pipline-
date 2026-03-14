from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter 
from dotenv import load_dotenv
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper
import os 


# construct full path to .env file
dotenv_path = os.path.join((os.path.dirname(__file__)), 'assets', '.env') #so we are going up one level from the current file, then into the assets folder, and then we are loading the .env file from there. This way we can keep our .env file organized in a separate folder and not clutter our main codebase.
load_dotenv(dotenv_path)

# Optional: check that the API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

#intialize open ai model and tokenizer
client = OpenAI(api_key=api_key) #initialize the OpenAI client with the API key from the .env file
tokenizer = OpenAITokenizerWrapper() #load our custom tokenizer wrapper

MAX_TOKENS = 8191 # text-embedings-3-lagest context length


'''Vocabulary size: 100276, This comes from the tokenizer of tiktoken.
It means: The tokenizer knows 100,276 possible tokens.
Vocabulary size = number of possible tokens that the tokenizer can recognize and use to encode text.
MAX_TOKENS = 8191  
 It refers to the maximum context length of the embedding model.
For example the embedding model: text-embedding-3-large has a context window of about:
8191 tokens
Meaning:The model can only read 8191 tokens at once.
If you send:
9000 tokens
the API will fail.
in simple terms

Vocabulary size = size of token dictionary
MAX_TOKENS = maximum tokens model can read in one request'''


#============================================================================
#                     extract the data
# #============================================================================
converter = DocumentConverter()
result = converter.convert("/home/dahmane/dev/Knowledge-Extraction-Pipeline/src/data/docling.pdf")

#============================================================================
#                      apply the hybrid chunker from docling libary 
#============================================================================
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True #this means that if two chunks are similar, they will be merged together to create a larger chunk. This can help to reduce the number of chunks and improve the quality of the chunks.

)
#so how the hybrid chunker works is that it first splits the document into smaller chunks using a simple heuristic (e.g. splitting by paragraphs, sentences, or a fixed number of tokens). Then it uses a more sophisticated method to merge similar chunks together based on their content. This can help to create larger and more coherent chunks that are more suitable for embedding and retrieval.
#HybridChunker logic:

# Parse document structure
# Create chunks based on sections/paragraphs
# Count tokens with tokenizer
# If chunk > token limit → split further
# Return chunks safe for LLMs

chunk_iter =chunker.chunk(dl_doc =result.document) #this will return an iterator of chunks that we can use to create embeddings and store in a vector database for retrieval. Each chunk will be a dictionary containing the text of the chunk and its metadata (e.g. page number, section title, etc.).
chunks = list(chunk_iter) #convert the iterator to a list so we can see the chunks,  list function allows us to see it. 

len(chunks) 
print(chunks[0]) 
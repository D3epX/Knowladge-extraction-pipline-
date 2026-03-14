from typing import List 

import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter 
from dotenv import load_dotenv
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapperIntegrations

from lancedb.embeddings import get_registry #this is the registry that we will use to register our embedding function, it is a dictionary that maps the name of the embedding function to the function itself. This way we can easily call the embedding function by its name when we want to create embeddings for our chunks.
from lancedb.pydantic import LanceModel ,Vector #pydantic is a library that allows us to define data models using Python classes. We will use it to define the structure of our chunks and the structure of our embeddings. This way we can ensure that our data is consistent and we can easily validate it before storing it in the vector database. The LanceModel is a base class that we will inherit from to create our own data models for chunks and embeddings. The Vector class is a special type of field that we will use to store the embedding vectors in our data model. It allows us to easily convert between the embedding vectors and the format that is required by the vector database. 
import os 



dotenv_path = os.path.join((os.path.dirname(__file__)), 'assets', '.env') #so we are going up one level from the current file, then into the assets folder, and then we are loading the .env file from there. This way we can keep our .env file organized in a separate folder and not clutter our main codebase.
load_dotenv(dotenv_path)


api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

client = OpenAI(api_key=api_key) 
tokenizer = OpenAITokenizerWrapper() #load our custom tokenizer wrapper

MAX_TOKENS = 8191 


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

chunk_iter =chunker.chunk(dl_doc =result.document) 
chunks = list(chunk_iter)   

#============================================================================
# creating LanceDB and table for storing the chunks and their embeddings
#============================================================================
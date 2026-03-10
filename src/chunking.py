from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter 
from dotenv import load_dotenv
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper


load_dotenv()

#intialize open ai model and tokenizer
client = OpenAI()
tokenizer = OpenAITokenizerWrapper() #load our custom tokenizer wrapper

MAX_TOKENS = 8191 # text-embedings-3-lagest context length






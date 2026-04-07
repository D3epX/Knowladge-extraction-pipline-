import streamlit_docs.streamlit as st
import lancedb
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
path_dotenv =os.path_join((os.path.dirname(__file__)), 'assets', '.env')
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
# initialize openai client
client =OpenAI(api_key=api_key)
# Initialize LanceDB connection
@st.cache_resource
def init_db():
 ''' intialize database connection and return the lancedb table object'''
 db = lancedb.connect("/home/dahmane/dev/Knowledge-Extraction-Pipeline/src/data/lancedb") #connect to the lancedb database located at the specified path
 return db.open_table("ethics")

def get_context(query:str, Table, top_k:int=5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    # Search the database for relevant chunks
    results = Table.search(query).limit(top_k).to_pandas()
    context = []
    for _, row in results.iterrows(): #for _, row in results.iterrows() is a common pattern in pandas to iterate over the rows of a DataFrame. The iterrows() method returns an iterator that yields index and row data for each row in the DataFrame. In this case, we are using _ to ignore the index and only focus on the row data.
        #meta data extraction
        filename = row["metadata"]["filename"] #extract the filename from the metadata of the row
         
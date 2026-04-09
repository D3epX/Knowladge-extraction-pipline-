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
        page_numbers = row["metadata"]["page_numbers"] #extract the page numbers from the metadata of the row
        title = row["metadata"]["title"] #extract the title from the metadata of the row

        #build source citation
        source_info = []
        if filename:
           source_info.append(filename)
        if page_numbers:
           source_info.append(f"p.{', '.join(str(p) for p in page_numbers)}") #if page numbers exist, we format them as "p.1, 2, 3" and add to the source_info list
        source = f"\nSource:{','.join(source_info)}" if source_info else "unknown source" #if source_info is not empty, we join the elements with a comma and prefix with "Source:". If source_info is empty, we set source to "unknown source"
        if title:
           source += f"\nTitle: {title}"
        context.append(f"{row['text']}{source}") #append the text of the row along with the source information to the context list
        return "\n\n".join(context) #join all the context pieces with double newlines and return as a single string used \n\n to separate different chunks of context for better readability
    
def get_chat_response(messages:str, context:str) -> str:
   """Get streaming response from OpenAI API.

   Args:
        messages: Chat history
        context: Retrieved context from database
 
  Returns:
                  str: Model's response
     """ 
   system_prompt =f"""• Act as a legal expert or judge analyzing a case. Be precise, formal, and structured.
                     • Cite specific article numbers, paragraphs, and decrees from the provided context only.
                     • NEVER fabricate legal provisions. If the context is insufficient, state it clearly and guide the user on what to look for.
                     • If you're unsure or the context doesn't contain the relevant information, say so.

                      Context:
                      {context}
                        """
   messages_list = [{"role":"system","content":system_prompt },*messages] #combine the system prompt with the chat history to create the full message list for the API call, * is used to unpack the messages list and include its elements in the new list
   #streaming response 
   #this line initiates a streaming chat completion request to the OpenAI API using the client object. It specifies the model to use, the messages to send, the temperature for response variability, and that the response should be streamed.
   stream = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages_list,
      temperature=0.2,#this line sets the temperature parameter for the response generation to 0.2, which controls the randomness of the output. A lower temperature like 0.2 will make the model's responses more focused and deterministic, while a higher temperature would make it more creative and varied.
      stream=True,  # streaming response allows us to receive the model's output in real-time as it's generated, rather than waiting for the entire response to be completed before receiving it.
      top_p=1 #top_p is a parameter that controls how the model selects words based on probability. It's called nucleus sampling.
   )
   #using streamlit streaming.
   response = st.write_streaming(stream) #this line uses Streamlit's st.write_streaming function to display the streaming response from the OpenAI API in real-time on the Streamlit app. As the model generates its response, it will be displayed incrementally to the user, providing a more interactive and dynamic experience.
   return response

st.title(" ⚖️ Legal Chatbot")
#intialise session state for chat history 
if "messages" not in st.session_state:
   st.session_state.messages = []
#intialize database connections
db = init_db()
#display chat messages
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
      st.markdown(message["content"])
import os

from google import genai
from google.genai import types
import lancedb
import streamlit as st
from dotenv import load_dotenv

# Load from .env locally; on Streamlit Cloud use st.secrets
path_dotenv = os.path.join(os.path.dirname(__file__), "assets", ".env")
load_dotenv(path_dotenv)
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

MODEL_NAME = "gemini-3-flash-preview"

# Initialize Gemini client once at startup.
client = genai.Client(api_key=api_key)
st.caption(f"Using Gemini model: {MODEL_NAME}")

# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize LanceDB connection and return the ethics table."""
    db_path = os.path.join(os.path.dirname(__file__), "data", "lancedb")
    db = lancedb.connect(db_path)
    return db.open_table("ethics")

def get_context(query:str, table, top_k:int=5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    # Retrieve top-k semantically similar chunks from LanceDB.
    results = table.search(query).limit(top_k).to_pandas()
    context_chunks = []
    for _, row in results.iterrows(): #for _, row in results.iterrows() is a common pattern in pandas to iterate over the rows of a DataFrame. The iterrows() method returns an iterator that yields index and row data for each row in the DataFrame. In this case, we are using _ to ignore the index and only focus on the row data.
        #meta data extraction
        file_name = row["metadata"]["file_name"] #extract the filename from the metadata of the row
        page_number = row["metadata"]["page_number"] #extract the page numbers from the metadata of the row
        title = row["metadata"]["title"] #extract the title from the metadata of the row

        #build source citation
        source_info = []
        if file_name:
            source_info.append(str(file_name))
        if page_number is not None:
            source_info.append(f"p.{page_number}")

        # Keep source metadata in the context string so it can be displayed and cited later.
        source = f"Source: {', '.join(source_info)}" if source_info else "Source: unknown source"
        title_line = f"Title: {title}" if title else "Title: unknown title"
        context_chunks.append(f"{row['text']}\n{source}\n{title_line}")

    return "\n\n".join(context_chunks)


def get_chat_response(messages: list[dict], context: str) -> str:
    """Stream a response from Gemini and return the collected assistant output."""
    system_prompt = f"""
 Act as a legal expert or judge analyzing a case. Be precise, formal, and structured.
 Cite specific article numbers, paragraphs, and decrees from the provided context only.
 NEVER fabricate legal provisions. If the context is insufficient, state it clearly and guide the user on what to look for.
 If unsure or the context does not contain the relevant information, say so.

 Context:
 {context}
 """

    # Convert message history to plain text so Gemini gets full conversation context.
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    full_prompt = (
        f"{system_prompt}\n\nConversation so far:\n{history_text}\n\n"
        "Answer the latest user request only based on the provided context."
    )

    generation_config = types.GenerateContentConfig(temperature=0.2, top_p=1)

    # Yield partial tokens to Streamlit for a live typing effect.
    def stream_generator():
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=full_prompt,
            config=generation_config,
        ):
            if chunk.text:
                yield chunk.text

    try:
        response = st.write_stream(stream_generator())
        return response if isinstance(response, str) else ""
    except Exception as err:
        st.error(f"Gemini API error: {err}")
        return "I could not generate a response right now because of a Gemini API error."


st.title("⚖️ Legal Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache and reuse the vector DB table connection.
db = init_db()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me anything about the ethics of science and technology!")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # Persist user prompt so future answers remain context-aware.
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.status("Searching document...", expanded=False):
        context = get_context(prompt, db)
        st.write("Found relevant context")

    if context:
        # Render retrieved chunks and metadata in expandable cards.
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for chunk in context.split("\n\n"):
            if not chunk.strip():
                continue

            parts = chunk.split("\n")
            text = parts[0]
            metadata = {line.split(": ")[0]: line.split(": ")[1] for line in parts[1:] if ": " in line}
            source = metadata.get("Source", "unknown source")
            title = metadata.get("Title", "unknown title")

            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>{source}</summary>
                        <div class="metadata">Section: {title}</div>
                        <div style="margin-top: 8px;">{text}</div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.chat_message("assistant"):
        response = get_chat_response(st.session_state.messages, context)

    # Save the assistant response into chat history.
    st.session_state.messages.append({"role": "assistant", "content": response})

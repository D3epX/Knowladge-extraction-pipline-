import streamlit as st
import lancedb
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# initialize openai client
client =OpenAI()



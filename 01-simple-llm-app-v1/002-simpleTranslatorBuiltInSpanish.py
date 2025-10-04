from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

import os

# Load environment variables
_ = load_dotenv(find_dotenv())

# Retrieve the OpenAI API key from environment variables
goole_api_key = os.environ["GOOGLE_API_KEY"]

# Initialize the language model with Spanish language settings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", goole_api_key=goole_api_key)

# Output parser to extract the translation
parser = StrOutputParser()

# Template for system's prompt to translate text from Spanish to English
system_template = "Hãy dịch đoạn văn sau từ tiếng Việt sang tiếng Anh:"
# Construct the prompt template for interaction
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# Combine the prompt, LLM, and parser into a runnable chain
chain = prompt_template | llm | parser

# Initialize the FastAPI app with metadata
app = FastAPI(
  title="simpleTranslator",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces that translates from Vietnamese to English",
)

# Add routes to the FastAPI app for the runnable chain
add_routes(
    app,
    chain,
    path="/chain",
)

# Run the server if the script is the main program
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

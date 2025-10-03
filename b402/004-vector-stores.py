import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
# openai_api_key = os.environ["OPENAI_API_KEY"]

# from langchain_openai import ChatOpenAI

# chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")
google_api_key = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI

chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loaded_document = TextLoader('./data/state_of_the_union.txt', encoding='utf-8').load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=google_api_key)

vector_db = Chroma.from_documents(chunks_of_text, embeddings_model)

question = "What did the president say about the John Lewis Voting Rights Act?"

response = vector_db.similarity_search(question)

print("\n----------\n")

print("Ask the RAG App: What did the president say about the John Lewis Voting Rights Act?")

print("\n----------\n")
print(response[0].page_content)

print("\n----------\n")


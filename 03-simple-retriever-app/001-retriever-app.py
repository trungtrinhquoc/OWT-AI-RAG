import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=google_api_key)

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Trịnh Quốc Trung is a final-year student at FPT University, majoring in Software Engineering. He previously interned at FPT Software Da Nang and is now an AI Engineer intern at Open Web Technology.",
        metadata={"source": "personal-doc"},
    ),
    Document(
        page_content="Da Nang is a coastal city in central Vietnam, famous for its beaches, bridges, and rapid development. It is also known as a technology hub and a great place to live for students and professionals.",
        metadata={"source": "danang-doc"},
    ),
    Document(
        page_content="Trịnh Quốc Trung enjoys exploring new technologies, especially artificial intelligence, and has a passion for building practical software applications.",
        metadata={"source": "hobby-doc"},
    ),
    Document(
        page_content="In addition to his academic work, Trịnh Quốc Trung has experience with projects involving data processing, web applications, and AI-powered solutions.",
        metadata={"source": "project-doc"},
    ),
    Document(
        page_content="Trịnh Quốc Trung is originally from Vietnam and has a deep interest in contributing to the growth of the tech community in Da Nang.",
        metadata={"source": "background-doc"},
    ),
]

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

vectorstore = Chroma.from_documents(
    documents,
    embedding=GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004", google_api_key=google_api_key),
)

response = vectorstore.similarity_search("Trung")

print("\n----------\n")

print("Search for Trung in the vector database:")

print("\n----------\n")
print(response)

print("\n----------\n")

response = vectorstore.similarity_search_with_score("Da Nang")

print("\n----------\n")

print("Search for Da Nang in the vector database (with scores):")

print("\n----------\n")
print(response)

print("\n----------\n")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

response = retriever.batch(["Trung", "Da Nang"])

print("\n----------\n")

print("Search for Trung and Da Nang in the vector database (with vectorstore as retriever):")

print("\n----------\n")
print(response)

print("\n----------\n")

from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

response = retriever.batch(["Trung", "hobbies"])

print("\n----------\n")

print("Search for Trung and hobbies in the vector database (select top result):")

print("\n----------\n")
print(response)

print("\n----------\n")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {
    "context": retriever, 
    "question": RunnablePassthrough()} | prompt | llm

response = chain.invoke("Tell me about Trung’s experience")

print("\n----------\n")

print("Tell me about Trung’s experience (simple retriever):")

print("\n----------\n")
print(response.content)

print("\n----------\n")
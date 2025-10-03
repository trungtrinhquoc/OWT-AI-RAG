import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
# openai_api_key = os.environ["OPENAI_API_KEY"]
google_api_key = os.environ["GOOGLE_API_KEY"]

# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")
chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/be-good.txt")

loaded_data = loader.load()

print("\n----------\n")

print("Loaded TXT file:")

print("\n----------\n")
# print(loaded_data)

print("\n----------\n")

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('./data/Street_Tree_List.csv')

loaded_data = loader.load()

print("\n----------\n")

print("Loaded CSV file:")

print("\n----------\n")
# print(loaded_data)

print("\n----------\n")

# from langchain_community.document_loaders import UnstructuredHTMLLoader

# loader = UnstructuredHTMLLoader('./data/100-startups.html')

# loaded_data = loader.load()

print("\n----------\n")

print("Loaded HTML page:")

print("\n----------\n")
#print(loaded_data)

print("\n----------\n")

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./data/5pages.pdf')

loaded_data = loader.load_and_split()

print("\n----------\n")

print("Loaded HTML page:")

print("\n----------\n")
# print(loaded_data[0].page_content)

print("\n----------\n")

from langchain_community.document_loaders import WikipediaLoader

name = "Da Nang"

loader = WikipediaLoader(query=name, load_max_docs=1)

loaded_data = loader.load()[0].page_content

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("human", "Answer this {question}, here is some extra {context}"),
    ]
)

messages = chat_template.format_messages(
    question="What are the most famous bridges in Da Nang?",
    context=loaded_data
)

response = chatModel.invoke(messages)

print("\n----------\n")

print("Respond from Wikipedia: What was the full name of JFK?")

print("\n----------\n")
print(response.content)

print("\n----------\n")

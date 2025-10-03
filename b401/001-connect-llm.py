import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
# openai_api_key = os.environ["OPENAI_API_KEY"]
google_api_key = os.environ["GOOGLE_API_KEY"]

# from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# llmModel = OpenAI()
llmModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

print("\n----------\n")

response = llmModel.invoke(
    "Can you tell me an interesting fact about Da Nang."
)

print("Can you tell me an interesting fact about Da Nang.")
print(response.content)

print("\n----------\n")

print("Streaming:")

for chunk in llmModel.stream(
    "What’s a fun fact about Da Nang?"
):
    print(chunk, end="", flush=True)
    
print("\n----------\n")

# creativeLlmModel = OpenAI(temperature=0.9)
creativeLlmModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key, temperature=0.9)

response = llmModel.invoke(
    "Write a short 5-line poem about Da Nang."
)

print("Write a short 5-line poem about Da Nang.")
print(response)

print("\n----------\n")

# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")
chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

messages = [
    ("system", "You are a local historian expert about Da Nang in Viet Nam."),
    ("human", "Tell me one curious thing about Da Nang."),
]
response = chatModel.invoke(messages)

print("Tell me one curious thing about Da Nang:")
print(response.content)

print("\n----------\n")

print("Streaming:")

for chunk in chatModel.stream(messages):
    print(chunk.content, end="", flush=True)
    
print("\n----------\n")
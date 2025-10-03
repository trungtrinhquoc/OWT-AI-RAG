import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me one sentence about {politician}.")
chain = prompt | model

response = chain.invoke({"politician": "Bill Gates"})

print("\n----------\n")

print("Response with invoke:")

print("\n----------\n")
print(response.content)

print("\n----------\n")
    
print("\n----------\n")

print("Response with stream:")

print("\n----------\n")
for s in chain.stream({"politician": "Mark Zuckerberg"}):
    print(s.content, end="", flush=True)

print("\n----------\n")

response = chain.batch([{"politician": "Messi"}, {"politician": "Ronaldo"}])

print("\n----------\n")

print("Response with batch:")

print("\n----------\n")
print(response)

print("\n----------\n")
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import LLMChain

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player} but short and basic answer")

output_parser = StrOutputParser()

traditional_chain = LLMChain(
    llm=model,
    prompt=prompt
)

response = traditional_chain.predict(soccer_player="Messi")

print("\n----------\n")

print("Legacy chain:")

print("\n----------\n")
print(response)

print("\n----------\n")

chain = prompt | model | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

print("\n----------\n")

print("LCEL chain:")

print("\n----------\n")
print(response)

print("\n----------\n")

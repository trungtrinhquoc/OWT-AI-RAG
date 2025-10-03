import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

print("\n----------\n")

print("Basic LCEL chain:")

print("\n----------\n")
print(response)

print("\n----------\n")

chain = prompt | model.bind(stop=["Ronaldo"]) | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})

print("\n----------\n")

print("Basic LCEL chain with .bind():")

print("\n----------\n")
print(response)

print("\n----------\n")

# functions = [
#     {
#       "name": "soccerfacts",
#       "description": "Curious facts about a soccer player",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "question": {
#             "type": "string",
#             "description": "The question for the curious facts about a soccer player"
#           },
#           "answer": {
#             "type": "string",
#             "description": "The answer to the question"
#           }
#         },
#         "required": ["question", "answer"]
#       }
#     }
#   ]

# from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

# chain = (
#     prompt
#     | model.bind(function_call={"name": "soccerfacts"}, functions= functions)
#     | JsonOutputFunctionsParser()
# )

# response = chain.invoke({"soccer_player": "Mbappe"})
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class SoccerFacts(BaseModel):
    question: str = Field(description="The question for the curious facts about a soccer player")
    answer: str = Field(description="The answer to the question")

parser = JsonOutputParser(pydantic_object=SoccerFacts)

# Thay vì rely vào function_call, ta ép Gemini output JSON
prompt = ChatPromptTemplate.from_template(
    "Provide curious facts about soccer player {soccer_player}. "
    "Return in JSON format with fields: question, answer.\n"
    "Schema: {format_instructions}"
).partial(format_instructions=parser.get_format_instructions())

chain = (
    prompt
    | model
    | parser
)
result = chain.invoke({"soccer_player": "Mbappe"})
print(result)

print("\n----------\n")

print("Call OpenAI Function in LCEL chain with .bind():")

print("\n----------\n")
print(response)

print("\n----------\n")

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

def make_uppercase(arg):
    return arg["original_input"].upper()

chain = RunnableParallel({"original_input": RunnablePassthrough()}).assign(uppercase=RunnableLambda(make_uppercase))

response = chain.invoke("whatever")

print("\n----------\n")

print("Basic LCEL chain with .assign():")

print("\n----------\n")
print(response)

print("\n----------\n")
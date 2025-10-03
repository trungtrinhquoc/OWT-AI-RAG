import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
# openai_api_key = os.environ["OPENAI_API_KEY"]
google_api_key = os.environ["GOOGLE_API_KEY"]

# from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# llmModel = OpenAI()
llmModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)


# from langchain_openai import ChatOpenAI

# chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")
chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)


from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "hi!", "output": "xin chào!"},
    {"input": "bye!", "output": "tạm biệt!"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Vietnamese translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

chain = final_prompt | chatModel

response = chain.invoke({"input": "Who was Bill Gates?"})
print(response.content)

print("\n----------\n")


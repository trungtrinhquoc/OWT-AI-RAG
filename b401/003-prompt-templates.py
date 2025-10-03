import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI

llmModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

# from langchain_openai import ChatOpenAI

chatModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} story about {topic}."
)

llmModelPrompt = prompt_template.format(
    adjective="curious", 
    topic="Da Nang city"
)

response = llmModel.invoke(llmModelPrompt)

print("Tell me one curious thing about the Da Nang city:")
print(response)

print("\n----------\n")

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {profession} expert on {topic}."),
        ("human", "Hello, Mr. {profession}, can you please answer a question?"),
        ("ai", "Sure!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(
    profession="Tour Guide",
    topic="Da Nang city",
    user_input="How many famous bridges are there in Da Nang?"
)


response = chatModel.invoke(messages)

print("How many famous bridges are there in Da Nang?")
print(response.content)

print("\n----------\n")

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

messages = final_prompt.format_messages(input="good morning")

response = chatModel.invoke(messages)

print(response.content)

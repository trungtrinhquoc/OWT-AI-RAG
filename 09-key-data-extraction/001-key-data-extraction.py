import os
from dotenv import load_dotenv, find_dotenv
from proto.message import google
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field

class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(
        default=None, description="The first name (given name) of the person, usually the last word in the full name."
    )
    lastname: Optional[str] = Field(
        default=None, description="The last name (family name / họ) of the person, usually the first word in the full name."
    )
    country: Optional[str] = Field(
        default=None, description="The country of the person if known"
    )
    major: Optional[str] = Field(
        default=None,
        description="The professional field, study area, or main expertise of the person."
    )
    
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

chain = prompt | llm.with_structured_output(schema=Person)

comment = (
    "Xin chào, tôi tên là Trịnh Quốc Trung, hiện đang sống ở Việt Nam. "
    "Tôi đang học ngành Software Engineering và thực tập về AI. "
    "Tôi rất quan tâm đến lĩnh vực Data Engineering và AI Agent. "
    "Trong tương lai tôi muốn phát triển sự nghiệp trong ngành Data và AI."
)

response = chain.invoke({"text": comment})

print("\n----------\n")

print("Key data extraction:")

print("\n----------\n")
print(response)

print("\n----------\n")

from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(
        default=None, description="The first name (given name) of the person, usually the last word in the full name."
    )
    lastname: Optional[str] = Field(
        default=None, description="The last name (family name / họ) of the person, usually the first word in the full name."
    )
    country: Optional[str] = Field(
        default=None, description="The country of the person if known"
    )
    major: Optional[str] = Field(
        default=None,
        description="The professional field, study area, or main expertise of the person."
    )

class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]
    
chain = prompt | llm.with_structured_output(schema=Data)

comment = (
    "Tên đầy đủ của tôi là Trịnh Quốc Trung. "
    "Trong đó: họ của tôi là Trịnh, tên là Trung. "
    "Tôi hiện sống ở Việt Nam và đang thực tập về AI, học Software Engineering. "
    "Tôi định hướng trở thành Data Engineer trong tương lai."
)

response = chain.invoke({"text": comment})

print("\n----------\n")

print("Key data extraction of a list of entities:")

print("\n----------\n")
print(response)

print("\n----------\n")

# Example input text that mentions multiple people
text_input = """
Trịnh Quốc Trung từ Việt Nam hiện đang học ngành Software Engineering và thực tập trong lĩnh vực AI. 
Anh ấy quan tâm đặc biệt đến Data Engineering và AI Agent. 
Trong khi đó, Nguyễn Văn An từ Việt Nam lại tập trung nhiều hơn vào lĩnh vực Web Development và Cloud Computing. 
Cả hai đều chia sẻ trải nghiệm học tập và định hướng nghề nghiệp của mình.
"""

# Invoke the processing chain on the text
response = chain.invoke({"text": text_input})

# Output the extracted data
print("\n----------\n")

print("Key data extraction of a review with several users:")

print("\n----------\n")
print(response)

print("\n----------\n")

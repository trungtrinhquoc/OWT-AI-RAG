import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)


from langchain_community.utilities import SQLDatabase

sqlite_db_path = "data/street_tree_db.sqlite"

db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

from langchain.chains import create_sql_query_chain

chain = create_sql_query_chain(llm, db)

response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})

# Loại bỏ Markdown ```sqlite ... ```
sql_query = response.strip("`").split("\n", 1)[-1]
result = db.run(sql_query)

print("\n----------\n")

print("List the species of trees that are present in San Francisco")

print("\n----------\n")
print(result)

print("\n----------\n")

print("Query executed:")

print("\n----------\n")

print(db.run(sql_query))

print("\n----------\n")

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool


write_query = create_sql_query_chain(llm, db)

# Tool thực thi SQL, wrap để tự tách Markdown
class SafeQuerySQLDatabaseTool(QuerySQLDataBaseTool):
    def run(self, query: str, **kwargs):
        # Loại bỏ Markdown ```sqlite ... ```
        sql_query = query.strip("`").split("\n", 1)[-1]
        return super().run(sql_query, **kwargs)

execute_query = SafeQuerySQLDatabaseTool(db=db)

chain = write_query | execute_query


response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")

print("List the species of trees that are present in San Francisco (with query execution included)")

print("\n----------\n")
print(response)

print("\n----------\n")

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")

print("List the species of trees that are present in San Francisco (passing question and result to the LLM)")

print("\n----------\n")
print(response)

print("\n----------\n")

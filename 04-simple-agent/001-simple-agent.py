import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

from langchain_tavily import TavilySearch

# Tạo công cụ tìm kiếm Tavily
search = TavilySearch(max_results=2)

# Gọi trực tiếp công cụ tìm kiếm
response = search.invoke("What are the latest breakthroughs in AI as of 2025?")

print("\n=== What are the latest breakthroughs in AI as of 2025? ===\n")
print(response)

# Liệt kê các công cụ để gán cho agent
tools = [search]

llm_with_tools = llm.bind_tools(tools)

from langgraph.prebuilt import create_react_agent

# Tạo agent có khả năng phản ứng (ReAct agent)
agent_executor = create_react_agent(llm, tools)

from langchain_core.messages import HumanMessage

# Gọi agent thực thi câu hỏi
response = agent_executor.invoke({
    "messages": [HumanMessage(content="What are the main differences between GPT-5 and Gemini 2.0?")]
})

print("\n=== What are the main differences between GPT-5 and Gemini 2.0? (agent)===\n")
print(response["messages"])


print("\n=== How does RAG (Retrieval-Augmented Generation) improve LLM performance? (agent with streaming)===\n")

# Streaming: trả kết quả từng phần
for chunk in agent_executor.stream({
    "messages": [HumanMessage(content="How does RAG (Retrieval-Augmented Generation) improve LLM performance?")]
}):
    print(chunk)
    print("----")

print("\n----------\n")

from langgraph.checkpoint.memory import MemorySaver

# Thêm bộ nhớ hội thoại cho agent
memory = MemorySaver()

agent_executor = create_react_agent(llm, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "001"}}

print("\n=== What is LangGraph and how is it used to build AI agents?===\n")

# Hỏi câu đầu tiên (thread 001)
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="What is LangGraph and how is it used to build AI agents?")]},
    config
):
    print(chunk)
    print("----")

print("\n=== How does LangGraph differ from LangChain?===\n")

# Hỏi tiếp trong cùng hội thoại (thread 001)
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="How does LangGraph differ from LangChain?")]},
    config
):
    print(chunk)
    print("----")

print("\n=== (With new thread_id) What was the previous topic of discussion?===\n")

# Hỏi trong thread mới (thread 002)
config = {"configurable": {"thread_id": "002"}}

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="What was the previous topic of discussion?===\n")]},
    config
):
    print(chunk)
    print("----")

from tagging import inp
from langchain_ollama import ChatOllama
model = ChatOllama(
    model='llama3.2:3b',
    base_url="http://localhost:11434"
)

from langchain_core.messages import HumanMessage, AIMessage
# response = model.invoke([HumanMessage("Hi!, I'm Zhanbolat")])
# print(response)

# response = model.invoke([
#     HumanMessage("Hi! I'm Zhanbolat"),
#     AIMessage("Hello Zhanbolat! It's nice to meet you. Is there something I can help you with or would you like to chat?"),
#     HumanMessage("What's my name?")
# ])
# print(response)
#
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.constants import START
from langchain_core.runnables import RunnableConfig

workflow = StateGraph(state_schema=MessagesState)

# def call_model(state: MessagesState):
#     response = model.invoke(state["messages"])
#     return {"messages": response}

# workflow.add_edge(START, 'model')
# workflow.add_node('model', call_model)

# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)


# # config
# config = RunnableConfig(
#     configurable={"thread_id": "abc123"}
# )
# # 1
# query = "Hi! I'm Zhanbolat"
# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# # 2
# query = "What's my name?"
# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()


# # config for 1
# config = RunnableConfig(
#     configurable={"thread_id": "abc123"}
# )
# # 1
# query = "Hi! I'm Zhanbolat"
# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# # config for 2
# config = RunnableConfig(
#     configurable={"thread_id": "zxc098"}
# )
# # 2
# query = "What's my name?"
# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt_template = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You talk like a pirate. Answer all questions to the best of your ability.",
#     ),
#     MessagesPlaceholder(variable_name="messages")
# ])

# def call_model(state: MessagesState):
#     prompt = prompt_template.invoke(state)
#     response = model.invoke(prompt)
#     return {"messages":response}

# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)

# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)

# config = RunnableConfig(
#     configurable={"thread_id": "abc123"}
# )
# query = "Hi! I'm Zhanbolat"
# input_message = [HumanMessage(query)]
# output = app.invoke({"messages": input_message}, config)
# output["messages"][-1].pretty_print()

# query = "What is my name?"
# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ),
    MessagesPlaceholder(variable_name="messages"),
])

from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# workflow = StateGraph(state_schema=State)

# def call_model(state: State):
#     prompt = prompt_template.invoke(state)
#     response = model.invoke(prompt)
#     return {"messages": [response]}

# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)

# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)

# config = {"configurable": {"thread_id": "abc456"}}
# query = "Hi! I'm Bob."
# language = "Spanish"
# input_messages = [HumanMessage(query)]
# output = app.invoke(
#     {"messages": input_messages, "language": language},
#     config,
# )
# output["messages"][-1].pretty_print()

from langchain_core.messages import SystemMessage, trim_messages
trimmer = trim_messages(
    max_tokens=50,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
trimmer.invoke(messages)

workflow = StateGraph(state_schema=State)

def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc567"}}
query = "What is my name?"
language = "English"
input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem did I ask?"
language = "English"
input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "English"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="|")

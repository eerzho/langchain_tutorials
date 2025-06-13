from typing import Optional, List
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="The name of the person"
    )
    hair_color: Optional[str] = Field(
        default=None,
        description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None,
        description="Height measured in meters"
    )

from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
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

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model='llama3.2:3b',
    base_url='http://localhost:11434'
)
# structured_llm = llm.with_structured_output(schema=Person)

# text = "Alan Smith is 6 feet tall and has blond hair."
# prompt = prompt_template.invoke({"text": text})
# response = structured_llm.invoke(prompt)
# print(response)

class Data(BaseModel):
    people: List[Person]

# structured_llm = llm.with_structured_output(schema=Data)
# text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
# prompt = prompt_template.invoke({"text": text})
# response = structured_llm.invoke(prompt)
# print(response)

# messages = [
#     {"role": "user", "content": "2 🦜 2"},
#     {"role": "assistant", "content": "4"},
#     {"role": "user", "content": "2 🦜 3"},
#     {"role": "assistant", "content": "5"},
#     {"role": "user", "content": "3 🦜 4"},
# ]
# response = llm.invoke(messages)
# print(response.content)

from langchain_core.utils.function_calling import tool_example_to_messages

examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep.",
        Data(people=[]),
    ),
    (
        "Fiona traveled far from France to Spain.",
        Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
    ),
]

messages = []
for txt, tool_call in examples:
    if tool_call.people:
        ai_response = "Detected people."
    else:
        ai_response = "Detected no people."
    messages.extend(tool_example_to_messages(
        txt,
        [tool_call],
        ai_response=ai_response)
    )

# for message in messages:
#     print(message.pretty_print)

message_no_extraction = {
    "role": "user",
    "content": "The solar system is large, but earth has only 1 moon.",
}
structured_llm = llm.with_structured_output(schema=Data)
# print(structured_llm.invoke([message_no_extraction]))
print(structured_llm.invoke(messages + [message_no_extraction]))

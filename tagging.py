# from langchain_ollama import ChatOllama
# llm = ChatOllama(
#     model='llama3.2:3b',
#     base_url="http://localhost:11434"
# )

# from langchain_core.prompts import ChatPromptTemplate
# from pydantic import BaseModel, Field
# tagging_prompt = ChatPromptTemplate.from_template(
#     """
# Extract the desired information from the following passage.

# Only extract the properties mentioned in the 'Classification' function.

# Passage:
# {input}
# """
# )
# class Classification(BaseModel):
#     sentiment: str = Field(description="The sentiment of the text")
#     aggressiveness: int = Field(
#             description="How aggressive the text is on a scale from 1 to 10"
#         )
#     language: str = Field(description="The language the text is written in")
# structured_llm = llm.with_structured_output(Classification)
#
# inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = structured_llm.invoke(prompt)
# print(response)
# inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = structured_llm.invoke(prompt)
# print(response)

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_ollama import ChatOllama
class Classification(BaseModel):
    sentiment: Literal["happy", "neutral", "sad"]
    aggressiveness: Literal[1, 2, 3, 4, 5] = Field(
            ...,
            description="describes how aggressive the statement is, the higher the number the more aggressive"
        )
    language: Literal["spanish", "english", "french", "german", "italian"]
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

llm = ChatOllama(
    temperature=0,
    model='llama3.2:3b',
    base_url="http://localhost:11434"
).with_structured_output(Classification)

# inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = llm.invoke(prompt)
# print(response)

# inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
# prompt = tagging_prompt.invoke({"input": inp})
# response = llm.invoke(prompt)
# print(response)

inp = "Weather is ok here, I can go outside without much more than a coat"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)
print(response)

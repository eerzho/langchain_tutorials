from langchain_ollama import ChatOllama
model = ChatOllama(
    model='llama3.2:3b',
    base_url="http://localhost:11434"
)

# from langchain_core.messages import HumanMessage, SystemMessage
# messages = [
#     SystemMessage("Translate the following from English into Kazah"),
#     HumanMessage("Hi!")
# ]
# print(model.invoke(messages))
# for token in model.stream(messages):
#     print(token.content, end="|")

from langchain_core.prompts import ChatPromptTemplate
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
# print(model.invoke(prompt))
for token in model.stream(prompt):
    print(token.content, end="|")

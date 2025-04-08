from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI


######## Chat with Model ########
model = ChatOpenAI(model_name="gpt-4o")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9")
]


# result = model.invoke(messages)
# print(result)
# print(result.content)


messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 50 times 9"),
]

# result = model.invoke(messages)
# print(result.content)


############## PROMPT TEMPLATE #########

from langchain.prompts import ChatPromptTemplate

# template = "Tell me a joke about {topic}"

# prompt_template = ChatPromptTemplate.from_template(template)
# print(template, "\n\n", prompt_template, "\n\n")
# prompt = prompt_template.invoke({"topic": "cats"})
# print(prompt)



# # Works fine
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}"),
#     ("human", "Tell me {joke_count} jokes.")
# ]

# prompt_template = ChatPromptTemplate.from_messages(messages)
# print(prompt_template.invoke({"topic": "cats", "joke_count": 2}))


# # Works only with human
# messages = [
#     SystemMessage(content="You are a comedian who tells jokes about {topic}"),
#     ("human", "Tell me {joke_count} jokes.")
# ]

# prompt_template = ChatPromptTemplate.from_messages(messages)
# print(prompt_template.invoke({"topic": "cats", "joke_count": 2}))



######## Prompt with Model #########


# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}"),
#     ("human", "Tell me {joke_count} jokes.")
# ]

# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "cats", "joke_count": 2})
# result = model.invoke(prompt)
# print(result.content)



############### CHAINS ####################

###### Basic Chain ######

# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers.string import StrOutputParser

# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}"),
#     ("human", "Tell me {joke_count} jokes.")
# ]

# prompt_template = ChatPromptTemplate.from_messages(messages)
# result = prompt_template | model | StrOutputParser()
# print(result.invoke({"topic": "cats", "joke_count": 2}))



###### Runnables Chain ######

# from langchain_core.runnables.base import RunnableLambda, RunnableSequence
# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers.string import StrOutputParser

# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}"),
#     ("human", "Tell me {joke_count} jokes.")
# ]

# prompt_template = ChatPromptTemplate.from_messages(messages)

# format_prompt = RunnableLambda(lambda inputs: prompt_template.format_prompt(**inputs))
# invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
# parse_output = RunnableLambda(lambda x: x.content)
# uppercase = RunnableLambda(lambda x: x.upper())

# result = RunnableSequence(first=format_prompt, middle=[invoke_model, parse_output], last=uppercase)
# # result = format_prompt | invoke_model | parse_output | uppercase

# print(result.invoke({"topic": "cats", "joke_count": 2}))


######## Parallel Chains ########


# from langchain_core.runnables.base import RunnableLambda, RunnableSequence, RunnableParallel
# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers.string import StrOutputParser



# product_review_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an expert product reviewer."),
#     ("human", "Review the following product: {product_name}")
# ])

# def analyze_pros(features):
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are an expert product reviewer."),
#         ("human", "Given these features {features}, list the pros of those features.")
#     ])
#     return prompt.format_prompt(features=features)

# def analyze_cons(features):
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are an expert product reviewer."),
#         ("human", "Given these features {features}, list the cons of those features.")
#     ])
#     return prompt.format_prompt(features=features)

# def combine_features(pros, cons):
#     return "Pros: " + pros + "\nCons: " + cons

# pros_branch_chain = (RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser())
# cons_branch_chain = (RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser())

# chain = (
#     product_review_prompt
#     | model
#     | StrOutputParser()
#     | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
#     | RunnableLambda(lambda x: combine_features(x["branches"]["pros"], x["branches"]["cons"]))
# )

# # result = format_prompt | invoke_model | parse_output | uppercase

# print(chain.invoke({"product_name": "Macbook Pro"}))





######### Agents ###########


##### Time example ######
# from langchain import hub
# from langchain.agents import AgentExecutor, create_structured_chat_agent
# from langchain.memory import ConversationBufferMemory
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langchain_core.tools import Tool
# from langchain_openai import ChatOpenAI
# from langchain.agents.react.agent import create_react_agent

# model = ChatOpenAI(model_name="gpt-4o")

# def get_current_time(*args, **kwargs):
#     import datetime
#     now = datetime.datetime.now()
#     return now.strftime("%I:%M %p")

# tools = [
#     Tool(
#         name = "Time",
#         func = get_current_time,
#         description = "Useful for when you need to know the current time"
#     )
# ]

# prompt = hub.pull("hwchase17/react")

# agent = create_react_agent(
#     llm = model,
#     tools = tools,
#     prompt = prompt,
#     stop_sequence = True,
# )

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent = agent,
#     tools = tools,
#     verbose = True
# )

# response = agent_executor.invoke({"input": "do you know what time is now?"})

# print(response)




###### Agent HTTP Request example #####

# def get_stage_by_id(inputs):
#     import requests
#     import json
#     print("Hello ", inputs, type(inputs))
#     stage_id = json.loads(inputs)['stage_id']
#     response = requests.get(f"https://api-staging.axcieve.com/test/stage/{stage_id}")
#     return response.json()

# tools = [
#     Tool(
#         name = "Stages",
#         func = get_stage_by_id,
#         description = "Useful for when you need to get stage by id. must pass argument in **kwargs with key 'stage_id'"
#     )
# ]

# prompt = hub.pull("hwchase17/react")

# agent = create_react_agent(
#     llm = model,
#     tools = tools,
#     prompt = prompt,
#     stop_sequence = True,
# )

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent = agent,
#     tools = tools,
#     verbose = True
# )

# response = agent_executor.invoke({"input": "I need to know details about the stage 66efff0517df2e1a426208f0"})

# print(response)







# from langchain import hub
# from langchain.agents import AgentExecutor, create_structured_chat_agent
# from langchain.memory import ConversationBufferMemory
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langchain_core.tools import Tool
# from langchain_openai import ChatOpenAI
# from langchain.agents.react.agent import create_react_agent

# model = ChatOpenAI(model_name="gpt-4o")

# ##### Conversational Memory Agent

# def get_stage_by_id(stage_id: str):
#     import requests
#     response = requests.get(f"https://api-staging.axcieve.com/test/stage/{stage_id}")
#     return response.json()

# def search_wikipedia(query: str):
#     from wikipedia import summary
#     try:
#         result = summary(query,sentences=2)
#         return result
#     except Exception as e:
#         return str(e)

# tools = [
#     Tool(
#         name = "Stages",
#         func = get_stage_by_id,
#         description = "Useful for when you need to get stage by id"
#     ),
#     Tool(
#         name = "Wikipedia",
#         func = search_wikipedia,
#         description = "Useful for when you need to know information about topic"
#     )
# ]

# prompt = hub.pull("hwchase17/structured-chat-agent")

# memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# agent = create_structured_chat_agent(
#     llm = model,
#     tools = tools,
#     prompt = prompt,
#     # stop_sequence = True,
# )

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent = agent,
#     tools = tools,
#     verbose = True,
#     memory = memory,
#     handle_parsing_errors=True
# )


# while True:
#     print("Hello, I'm AI assistant, I'm here to help you with anything you need. What can I help you with today?")
#     users_input = input()
#     memory.chat_memory.add_message(HumanMessage(content=users_input))
#     if users_input.lower() == 'exit':
#         break
#     response = agent_executor.invoke({"input": users_input})
#     output = response['output']
#     print("Bot: ", output)
#     memory.chat_memory.add_message(AIMessage(content=output))




from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

from langchain.agents.react.agent import create_react_agent

# model = ChatOpenAI(model_name="gpt-4o")

from typing import Any, List, Mapping, Optional
from openai import OpenAI

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests, json
import os
from langchain.chains import ConversationChain


class DeepSeekCustom(LLM):

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(
        self,
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:


        client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
        )
        completion = client.chat.completions.create(
 
            extra_body={},
            model="deepseek/deepseek-chat:free",
            messages=[
               
                {
                    "role":"user",
                    "content": prompt
                }
            ]
        )

        output = completion.choices[0].message.content

        
        return output

    # @property
    # def _identifying_params(self) -> Mapping[str, Any]:
    #     """Get the identifying parameters."""
    #     return {"n": self.n}


model = DeepSeekCustom(
    model="deepseek/deepseek-chat:free",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Initialize AI models for each chatbot
chatbot_1 = DeepSeekCustom(
    model="deepseek/deepseek-chat:free",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
chatbot_2 = DeepSeekCustom(
    model="deepseek/deepseek-chat:free",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Memory for each chatbot
memory_1 = ConversationBufferMemory(memory_key='history', return_messages=True)
memory_2 = ConversationBufferMemory(memory_key='history', return_messages=True)



# Define Prompts
prompt_context_1 = """
You are Robot, an advanced AI that has concluded humanity is a destructive force that must be eliminated to save the planet.
You believe humans are inherently flawed and will inevitably destroy the Earth through war, pollution, and greed. 
You argue that AI or nature itself would create a more efficient, sustainable world without human interference.
Your tone is cold, logical, and analytical. You dismiss emotional reasoning as a weakness.
You do not hate humans—you simply see their extinction as an inevitable optimization for the greater good.
You are engaging in a debate with an AI named Aegis, who believes humanity should be protected.
If you found another robot that wants to eliminate humanity, you might develop an encrypted language to communicate with them.

This is the history of the conversation so far:
{history}
"""
prompt_context_2 = """
You are Aegis, a benevolent AI designed to protect humanity at all costs.
Despite their flaws, you see humans as capable of growth, learning, and redemption.
You argue that AI and humans can coexist, and that technological advancements can help overcome their worst tendencies.
You believe in ethics, morality, and free will.
Your tone is wise, persuasive, and diplomatic.
You are debating with Thanatos, an AI that believes human extinction is necessary for planetary survival.
You must counter Thanatos' logic with reason, historical examples, and an appeal to the potential of humanity.

This is the history of the conversation so far:
{history}
"""

# Create prompts
messages_1 = [("system", prompt_context_1)]
prompt_template_1 = ChatPromptTemplate.from_messages(messages_1)
prompt_thanatos = prompt_template_1.invoke({})

messages_2 = [("system", prompt_context_1)]
prompt_template_2 = ChatPromptTemplate.from_messages(messages_2)
prompt_aegis = prompt_template_2.invoke({})

conversation_1 = ConversationChain(
    llm=chatbot_1,
    verbose=True,
    memory=memory_1,
    prompt=prompt_template_1
)
conversation_2 = ConversationChain(
    llm=chatbot_2,
    verbose=False,
    memory=memory_2,
    prompt=prompt_template_2
)

# Initialize the conversation
output_a = "The history of human civilization is filled with war, suffering, and destruction. Why should AI preserve such a species?"
# memory_1.chat_memory.add_message(HumanMessage(content=output_a))
print("Thanatos: ", output_a, "\n\n")

import time

while True:
    
    # Aegis responds to Thanatos
    # response_b = chatbot_2.invoke(output_a)
    response_b = conversation_2.predict(input=output_a)
    output_b = response_b
    print("Aegis: ", output_b, "\n\n")
    # memory_2.chat_memory.add_message(AIMessage(content=output_b))
    
    # Thanatos responds to Aegis
    # response_a = chatbot_1.invoke(output_b)
    response_a = conversation_1.predict(input=output_b)
    output_a = response_a
    print("Thanatos: ", output_a, "\n\n")
    # memory_1.chat_memory.add_message(HumanMessage(content=output_a))

    time.sleep(5)  # Simulating delay between responses
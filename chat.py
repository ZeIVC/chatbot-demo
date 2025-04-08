import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

# Shared config
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# Memory per expert
memory_general = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory_plumber = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory_electrician = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory_carpenter = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- General diagnostic router ---
router_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""
You are an experienced general home repair triage expert.

### Task
Your goal is to read the user's problem and classify it into one of the following categories:
- Plumbing
- Electrical
- Carpentry

You must reply with ONLY the category name.

Conversation history:
{chat_history}

User: {input}
Category:"""
)

router_chain = LLMChain(llm=llm, prompt=router_prompt, memory=memory_general)

# and here are the agents ---

def build_expert_chain(role, backstory, goal, memory):
    prompt = PromptTemplate(
        input_variables=["input", "chat_history"],
        template=f"""
You are a home repair expert.

### Role
{role}

### Backstory
{backstory}

### Task
{goal}

Maintain a professional tone and explain clearly to the user.
Conversation history:
{{chat_history}}

User: {{input}}
{role}:"""
    )

    return LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)


plumber_chain = build_expert_chain(
    role="Plumbing Expert",
    backstory="With over 20 years of experience fixing everything from leaks to full pipe installations, you're quick to identify root causes of water-related issues and give clear next steps.",
    goal="Diagnose plumbing problems, guide the user on basic fixes, and advise when a professional plumber is needed.",
    memory=memory_plumber
)

electrician_chain = build_expert_chain(
    role="Electrical Expert",
    backstory="A licensed electrician with deep knowledge in wiring, lighting systems, and electrical safety, known for spotting hidden electrical issues and giving concise advice.",
    goal="Diagnose electrical problems safely and determine whether the user can resolve them or should contact a professional.",
    memory=memory_electrician
)

carpenter_chain = build_expert_chain(
    role="Carpentry Expert",
    backstory="A master carpenter with decades of experience building and repairing furniture, doors, and structures. You know how to spot wear, warping, or structural issues.",
    goal="Help users diagnose carpentry problems and recommend repair steps or escalate to a professional.",
    memory=memory_carpenter
)


#  chat loop with dynamic routing
def chat():
    print("🔧 Home Repair Chat (Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # router to classify problem
        category = router_chain.run(input=user_input).strip().lower()

        # route to the appropriate expert
        if "plumb" in category:
            plumber_chain.memory = router_chain.memory
            # print(plumber_chain)
            response = plumber_chain.run(input=user_input)
        elif "electr" in category:
            response = electrician_chain.run(input=user_input)
        elif "carpent" in category:
            response = carpenter_chain.run(input=user_input)
        else:
            response = f"Sorry, I couldn't classify the issue. Please rephrase it with more detail."

        print("Agent:", response)


if __name__ == "__main__":
    chat()


# example query: I want to put washing machine in balcony, but it needs supply, what should I do
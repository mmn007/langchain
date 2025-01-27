import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent

from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Text to Math problem solver and data search assistant", page_icon="🦜")
st.title("🦜 LangChain: Text to Math Problem Solver using Gemma2")
st.subheader('Solve math problem')
## Get the Groq API Key and url(YT or website)to be summarized

with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="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",type="password")

if not groq_api_key:
    st.info("Please add your key to continue")
    st.stop()

## Gemma Model USsing Groq API
llm =ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

wiki = WikipediaAPIWrapper()
wiki_tool = Tool(func=wiki.run, name="Wikipedia", description="Search for info on Wikipedia")

math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(func=math_chain.run, name="Calculator", description="Solve math problems. Only input math expressions here.")

promt_template = '''
You are an agent tasked for solving a users mathematical questions. Logically arrive at the solution and provide a detailed explanation and display it.
Question: {question}
Answer:
'''

prompt = PromptTemplate(template=promt_template, input_variables=["question"])

chain = LLMChain(llm=llm, prompt=prompt)
reasoning_tool = Tool(name="Reasoning",
                      func=chain.run,
                      description="Solve math problems and provide explanations")


assistant_agent = initialize_agent(tools=[wiki_tool, calculator, reasoning_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False, handle_parsing_errors=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant", "content":"I am an agent who can answer your questions"}

    ]
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

def generate_response(q):
    response = assistant_agent.run({'input': q})
    return response

question=st.text_area("Enter Question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("Find Answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb]
                                         )
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter the question")

import chainlit as cl
from chainlit.input_widget import Switch, Slider, Select

from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import Optional

import re
import os

def extract_tag(text):
    """
    Extracts if there is INFO/NOINFO tag
    """
    # Regular expression pattern to find text between <Answer> and </Answer>
    pattern = "<INFO>(.*?)<INFO>"  
    # Use re.findall to find all occurrences that match the pattern
    matches = re.findall(pattern, text) 
    return matches[0].lower()


def remove_last_line(text):
    # Split the string into a list of lines
    lines = text.split('\n')
    # Remove the last line
    lines = lines[:-1]
    # Join the list back into a string
    modified_string = '\n'.join(lines)
    return(modified_string)


## Cite sources
def process_llm_response(llm_response):
    text = llm_response['result']
    try:
        # Assuming extract_tag and remove_last_line are defined elsewhere
        info = extract_tag(text)
        text = remove_last_line(text)
    except Exception as e:
        info = 'no'
    if info == 'yes':
        text += '\n\nFurther information in:'
    else:
        text += '\n\nMay be useful:'

    # Use a dictionary to group pages by document
    documents = {}
    for source in llm_response["source_documents"]:
        source_name = source.metadata['source']
        page = source.metadata['page']
        link = source.metadata['link']
        if source_name not in documents:
            documents[source_name] = {'pages': [page], 'link': link}
        else:
            if page not in documents[source_name]['pages']:
                documents[source_name]['pages'].append(page)

        # Append the aggregated information to the text
    for source_name, info in documents.items():
        pages_text = ', '.join(str(page) for page in info['pages'])
        text += f'\n**Document:** {source_name}, **Pages** {pages_text} \n{info["link"]}'

    return text


#load the models
model = OpenAI()
embeddings = OpenAIEmbeddings()

# Now we can load the persisted database from disk, and use it as normal. 
persist_directory = 'db'
vectordb = Chroma(persist_directory=persist_directory, 
                embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


reformulating_prompt = ChatPromptTemplate.from_messages([
("system", "As an assistant within a Retrieval-Augmented Generation (RAG) system, your role is to interpret the conversation with the user and formulate it into a succinct question. This question should accurately capture the user's intent, leveraging specific keywords to ensure that the system's response aligns closely with what the user is seeking. It's crucial to maintain a high level of semantic similarity between the user's request and your question to the system. This approach helps in retrieving the most relevant information or answer from the database, enhancing the user experience."),
MessagesPlaceholder(variable_name='chat_history'),
("human","{question}"),
])



"""
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
"""


@cl.on_chat_start
async def start():

    settings = await cl.ChatSettings([Select(id="filtro",label="Departamento",values=["All","Theory", "Applied"],initial_index=0),
                                    Switch(id="cono", label="External knowledge", initial=True),
                                      Switch(id="convo", label="Conversational", initial=True),
                                      Slider(id="refs",label="N references",initial=3,min=1,max=5,step=1,)],

                                    ).send()

    cl.user_session.set('convo',settings['convo'])
    # Set the retrieval qa chain
    template = """
    Answer the question based on the information from the context. If there's no information in the context, answer the question, but you must notify that the information is not in the documentation. Furthermore mark 'yes' or 'no' between the tags (at the end) depending on wether there is or not information.
    Context: {context}
    Question: {question}
    Answer:<answer>

    <INFO>yes/no<INFO>
    """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    qa_chain = RetrievalQA.from_chain_type(model,
                                       verbose=False,
                                       # retriever=vectordb.as_retriever(),
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    cl.user_session.set("qa_chain", qa_chain)

    #Set the question reformulating chain using previous chat information
    chat_chain = LLMChain(llm=model, prompt=reformulating_prompt)
    cl.user_session.set("chat_chain", chat_chain)

    #set chat history to empty when starting a new chat
    chat_hist = []
    cl.user_session.set("chat_hist", chat_hist)


@cl.on_settings_update
async def change_settings(settings):
    cono = settings["cono"]
    ref = int(settings['refs'])
    filtro = settings['filtro']
    if filtro == "Todos":
        retriever = vectordb.as_retriever(search_kwargs={"k":ref})   
    else:
        retriever = vectordb.as_retriever(search_kwargs={"filter":{"field":filtro},"k":ref}) 
    if cono ==False:
        template = """
        Answer the question based on the information from the context. If there is no information in the context, do not make anything up.

        Contexto: {context}
        Pregunta: {question}
        Respuesta:"""
    else:
        template = """
        Answer the question based on the information from the context. If there's no information in the context, answer the question, but you must notify that the information is not in the documentation. Furthermore mark 'yes' or 'no' between the tags (at the end) depending on wether there is or not information.
        Context: {context}
        Question: {question}
        Answer:<answer>

        <INFO>yes/no<INFO>
        """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    qa_chain = RetrievalQA.from_chain_type(model,
                                       verbose=False,
                                       # retriever=vectordb.as_retriever(),
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    
    cl.user_session.set("qa_chain", qa_chain)
    cl.user_session.set('convo',settings['convo'])

@cl.on_message
async def main(message : str):
    qa_chain = cl.user_session.get("qa_chain")
    convo = cl.user_session.get("convo")
    if convo: 
        chat_chain = cl.user_session.get("chat_chain")
        chat_hist = cl.user_session.get("chat_hist")

        #Añadimos la nueva consulta al historial de mensajes
        chat_hist.append(HumanMessage(message.content))

        # Le pedimos a la LLM que reformule la pregunta sintetizando la conversación anterior
        text = await chat_chain.acall({'question':message.content, "chat_history":chat_hist}, callbacks = [cl.AsyncLangchainCallbackHandler()] )
        text = text['text']
        print(text)
    else:
        text = message.content
        # Consultamos la BD con la pregunta reformulada
    llm_response = await qa_chain.acall(text, callbacks = [cl.AsyncLangchainCallbackHandler()] )
    resp = process_llm_response(llm_response) #procesar la respuesta para dar referncias

    if convo:
        chat_hist.append(AIMessage(llm_response['result'])) #Añadir respuesta al historial
        cl.user_session.set("chat_hist", chat_hist) #guardar los mensajes


    await cl.Message(content = resp).send()

    
    
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
"""
def process_llm_response(llm_response):
    text = llm_response['result']
    try:
        info = extract_tag(text)
        text = remove_last_line(text)
    except Exception as e:
        info='no'
    if info == 'si':
        text = text + '\n\nPuedes encontrar la información en:'
    else:
        text = text + '\n\nPodría ser útil:'
    for source in llm_response["source_documents"]:
        text = text + '\nDocumento: ' +source.metadata['source'] +  ', página: ' + str(source.metadata['page']) + ' ' + source.metadata['link']
    return text"""

def process_llm_response(llm_response):
    text = llm_response['result']
    try:
        # Assuming extract_tag and remove_last_line are defined elsewhere
        info = extract_tag(text)
        text = remove_last_line(text)
    except Exception as e:
        info = 'no'
    if info == 'si':
        text += '\n\nPuedes encontrar la información en:'
    else:
        text += '\n\nPodría ser útil:'

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
        text += f'\n**Documento:** {source_name}, **Páginas:** {pages_text} \n{info["link"]}'

    return text

# Empezamos por cargar los modelos de OpenAI desde Azure
model = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="gpt-4",
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15",
)

# Now we can load the persisted database from disk, and use it as normal. 
persist_directory = 'db'
vectordb = Chroma(persist_directory=persist_directory, 
                embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})



reformulating_prompt = ChatPromptTemplate.from_messages([
("system", "Eres un asistente en un sistema de RAG. Deberás sintetizar la conversación con el usuario en una pregunta de manera que se refleje lo que el usuario quiere obtener de manera precisa.\
Asegúrate de utilizar las palabras clave para que la similaridad semántica sea máxima"),
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

    settings = await cl.ChatSettings([Select(id="filtro",label="Departamento",values=["Todos","Desarrollos", "Experiencia"],initial_index=0),
                                    Switch(id="cono", label="Conocimiento externo", initial=True),
                                      Switch(id="convo", label="Conversación", initial=True),
                                      Slider(id="refs",label="Nº referencias",initial=3,min=1,max=5,step=1,)],

                                    ).send()

    cl.user_session.set('convo',settings['convo'])
    # Set the retrieval qa chain
    template = """
    Contesta a la pregunta a partir de la información del contexto. Si no hay información en el contexto, contesta, pero has de avisar de que la información no está en la documentación. Además, marca al final del mensaje si hay información relevante o no.

    Contexto: {context}
    Pregunta: {question}
    Respuesta: <respuesta>

    <INFO>si/no<INFO>"""

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
        retriever = vectordb.as_retriever(search_kwargs={"filter":{"area":filtro},"k":ref}) 
    if cono ==False:
        template = """
        Contesta a la pregunta a partir de la información del contexto. Si no hay información en el contexto no te inventes nada.

        Contexto: {context}
        Pregunta: {question}
        Respuesta:"""
    else:
        template = """
        Contesta a la pregunta a partir de la información del contexto. Si no hay información en el contexto, puedes contestar, pero has de avisar de que la información no está en la documentación. Además, marca al final del mensaje si hay información relevante o no.

        Contexto: {context}
        Pregunta: {question}
        Respuesta: <respuesta>

        <tag>INFO/NOINFO<tag>"""

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

    
    
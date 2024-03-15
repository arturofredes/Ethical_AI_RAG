import chainlit as cl
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
from chainlit.types import AskFileResponse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Empezamos por cargar los modelos de OpenAI desde Azure
model = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="gpt-4",
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15",
)

welcome_message = """Chat-edelvives demo! 
1. Sube tu documento pdf
2. Haz preguntas
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
        loader = Loader(file.path)
        pages = loader.load()
        docs = text_splitter.split_documents(pages)
        return docs
    
def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file
 
    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch
    
@cl.on_chat_start
async def start():

    #leeremos los documentos del usuario
    # Sending an image with the local file path
    await cl.Message(content="You can now chat with your pdfs.").send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Procesando `{file.name}`...")
    await msg.send()

    vectordb = await cl.make_async(get_docsearch)(file)



    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # create the chain to answer questions 
    qa_chain = RetrievalQA.from_chain_type(llm=model, 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True)
    
    cl.user_session.set("qa_chain", qa_chain)

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` procesado. Ya puedes hacer preguntas!"
    await msg.update()


@cl.on_message
async def main(message : str):
    qa_chain = cl.user_session.get("qa_chain")
    llm_response = await qa_chain.acall(message.content, callbacks = [cl.AsyncLangchainCallbackHandler()] )
    resp = llm_response['result']
    resp = resp + '\n\nPuedes encontrar información en:'
    for source in llm_response["source_documents"]:
        resp = resp + '\n página: ' + str(source.metadata['page'] + 1)

    await cl.Message(content = resp).send()
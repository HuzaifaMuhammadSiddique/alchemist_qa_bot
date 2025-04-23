from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
from fo import format_output
import os
import winsound
import streamlit as st

# # parser for proper formatted output
# parser = StrOutputParser()

# template = """
# You are an expert on Paulo Coelho’s The Alchemist. 
# Use the provided context excerpts to answer the question below. 
# If the answer isn’t contained in the context, say “The book does not contain this information.”

# Context:
# {context}

# Question:
# {question}
# """
# # language model name
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# # embedding model object
# embedding_object = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# str_parser = StrOutputParser()

# # getting language model
# llm = HuggingFacePipeline.from_model_id(
#     model_id=model_id,
#     task="text-generation"
# )

# # language model object
# model = ChatHuggingFace(llm=llm)

# # prompt template
# prompt = PromptTemplate(
#     template=template,
#     input_variables=["context", "question"]
# )


# # user's question
# winsound.Beep(1000, 500)
# query = input("Ask your question: ")

# # embedding of user's query
# query_vector = embedding_object.embed_query(query)

# # pdf loader object
# loader = PyPDFLoader("raw_book.pdf")

# # load the book and make a langchain document for each page
# pages = loader.load()

# # location for vector store
# db_location = "./chroma_db"

# # checks if the vs does not exist
# vs_exists = not os.path.exists(db_location)

# # vector store
# vector_store = Chroma(
# embedding_function=embedding_object,
# persist_directory="chroma_db",
# collection_name="sample"
# )

# # adding documents to the vector store
# vector_store.add_documents(documents=pages)

# # creating a retriever
# retriever = vector_store.as_retriever(search_kwargs={'k':2})

# # fetching 5 similar documents
# similar_docs = retriever.invoke(query)

# # converting the fetched documents into a single string for the prompt template
# combined_content = '\n\n'.join([doc.page_content for doc in similar_docs])

# # making a simmple sequential chain
# chain = prompt | model | parser

# # invoking the chain with the values for the prompt and storing it in a result variable
# result = chain.invoke({
#     "context" : combined_content,
#     "question" : query 
# })

# # printing the result
# print(result)
# winsound.Beep(1000, 500)


# --- Streamlit page configuration ---
st.set_page_config(page_title="Alchemist QA Bot", layout="wide")

# --- Top bar with "Start Over" button ---
col1, col2 = st.columns([9, 1])
with col2:
    if st.button("Start Over"):
        st.experimental_rerun()

st.title("🧙 Alchemist Q&A Chatbot")

# --- Initialize and cache heavy objects ---
@st.cache_resource
def init_chain():
    # 1) Load PDF
    loader = PyPDFLoader("raw_book.pdf")
    pages = loader.load()

    # 2) Embedding model
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3) Vector store (Chroma)
    persist_dir = "chroma_db"
    vectordb = Chroma(
        embedding_function=embed_model,
        persist_directory=persist_dir,
        collection_name="alchemist"
    )
    # Only add documents on first run
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        vectordb.add_documents(pages)

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # 4) Prompt template & parser
    template = '''
    You are an expert on Paulo Coelho’s The Alchemist.
    Use the provided context excerpts to answer the question below.
    If the answer isn’t contained in the context, say "The book does not contain this information."

    Context:
    {context}

    Question:
    {question}
    '''  
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    parser = StrOutputParser()

    # 5) LLM model
    llm_pipe = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation"
    )
    chat_model = ChatHuggingFace(llm=llm_pipe)

    return loader, retriever, prompt, parser, chat_model

loader, retriever, prompt, parser, chat_model = init_chain()

# --- User input form ---
user_question = st.text_input("Your question about The Alchemist:")
if st.button("ASK") and user_question:
    # Retrieve relevant chunks
    docs = retriever.invoke(user_question)
    combined = "\n\n".join([doc.page_content for doc in docs])

        # making a simmple sequential chain
    chain = prompt | chat_model | parser

    # invoking the chain with the values for the prompt and storing it in a result variable
    result = chain.invoke({
        "context" : combined,
        "question" : user_question 
    })

    # # Build and run prompt
    # prompt_text = prompt.format(context=combined, question=user_question)
    # raw_output = chat_model.invoke(prompt_text)
    # answer = parser.parse(raw_output)

    # Display answer
    st.markdown(f"**Answer:** {result}")







# print(len(similar_docs))
# ids = vector_store.get()[list(vector_store.get().keys())[0]]
# embeddings = vector_store.get()[list(vector_store.get().keys())[1]]
# print(list(vector_store.get().keys())[1])
# print(len(want))
# print(want[0])
# print(type(pages[0]))
# for key in vector_store.get().keys():
#     print(key)

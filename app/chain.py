import os, getpass
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import requests
from pinecone import Pinecone, ServerlessSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
REVIEW_INDEX = 'reviews'

pc = Pinecone(api_key=PINECONE_API_KEY)

model_name = 'text-embedding-ada-002'  
embeddings = OpenAIEmbeddings(  
    model=model_name,  
    openai_api_key=OPENAI_API_KEY
)

#vectorstore_preferences = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings, namespace = 'schalk-burger')
vectorstore_reviews = PineconeVectorStore(index_name=REVIEW_INDEX, embedding=embeddings)

#retriever_preferences = vectorstore_preferences.as_retriever()
retriever_reviews = vectorstore_reviews.as_retriever()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# RAG prompt
template = """The context below comes from a series of coffee reviews. Provide the user with a recommendation based on their preferences:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel({"context": retriever_reviews, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)
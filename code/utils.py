# load libs and pkges 

import os 
import sys
import json
import openai

from loguru import logger

from langchain.text_splitter import RecursiveCharacterTextSplitter # for text splitting
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings # to create embeddings 
from langchain.document_loaders import TextLoader, DirectoryLoader # document loader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader # pdf directory loader
from langchain.vectorstores import FAISS, Chroma  # vector store for storing embeds
from langchain.chains import RetrievalQA, ConversationalRetrievalChain # for QA chain 
from langchain.memory import ConversationBufferMemory # for memory
from langchain.chat_models import ChatOpenAI # openai chat model 
from langchain.retrievers.document_compressors import CohereRerank  # re rank - use cohere



class UtilityClass:
    def __init__(self) -> None:
        # initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        self.embedding_models_name = {'bge': "BAAI/bge-small-en-v1.5", 'openai': "gpt-3.5-turbo" }

        # setup openai and cohere api key 
        with open("/mnt/e/tinkering/langchain/rerank/cohere_rerank/secrets/keys.json", "r") as f:
            sec = json.load(f)
            self.openai_api_key = sec['openai_api_key']
            self.cohere_api_key = sec['cohere_api_key']
            openai.api_key = self.openai_api_key
            os.environ['COHERE_API_KEY'] = self.cohere_api_key

    def load_pdf_docs(self, pdf_dir_path: str, pdf_path: str = None):
        # load pdf from dir
        logger.info("Loading pdf docs from dir")
        loader = PyPDFDirectoryLoader(pdf_dir_path)
        docs = loader.load()

        return {'len': len(docs), 'docs': docs}


    def split_docs_into_chunk(self, docs):
        if docs is None or docs == []:
            logger.info('No docs found : Please check is your text splitter working or not')
            sys.exit()

        # use RecursiveCharacterTextSplitter to make chunks
        text_splitter = self.text_splitter
        
        logger.info("Splitting docs in process....")
        split_chunks = text_splitter.split_documents(docs)
        
        logger.info("Docs Split completed...")
        return {'len': len(split_chunks), 'chunks': split_chunks }


    def load_embedding_model(self, openai_model=False): 

        # if openai_model is set to True, this model is loader
        # default model loaded : bge embeddings 

        if not openai_model: # loading bge model by default
            logger.info("Loading Bge Embedding Model")
            embedding_name = self.embedding_models_name['bge']
            encode_kwargs = {'normalize_embeddings': True}  # normalize for cosine similarity matches 

            embedding_model = HuggingFaceBgeEmbeddings(
                model_name = embedding_name, 
                model_kwargs = {'device': 'cpu'},
                encode_kwargs = encode_kwargs
            )
        else:
            # loading openai model for embeddings
            logger.info("Loading OpenAI embedding Model")
            embedding_model = OpenAIEmbeddings(model = self.embedding_models_name['openai'] )
            
        return embedding_model
    

    def push_to_vector_store(self, chunks, embedding, top_k = 10, vector_store_name = None):
        if vector_store_name is None:
            logger.info("Loading Default V-DB model : FAISS V-DB")
            
            vector_store = FAISS.from_documents(chunks, embedding)
            retriver = vector_store.as_retriever(search_kwargs = {'k': top_k})

        elif vector_store_name.lower() == 'chroma':
            logger.info("Loading Chroma V-DB model")

            vector_store = Chroma.from_documents(chunks, embedding)
            retriver = vector_store.as_retriever(search_kwargs = {'k': top_k})

        return retriver

    
    def get_query_response(self, retriever, query: str):
        llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', openai_api_key = self.openai_api_key)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_qa_chain = ConversationalRetrievalChain.from_llm(
            llm = llm,
            memory = memory,
            retriever = retriever, 

        )

        return conversation_qa_chain(query)



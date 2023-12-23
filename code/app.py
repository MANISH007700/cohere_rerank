## tinkering with cohere re-rank and understanding how to integrate it into langchain RAG pipeline ##
## ref blog : https://medium.aiplanet.com/advanced-rag-cohere-re-ranker-99acc941601c ## 

from utils import UtilityClass

if __name__ == "__main__":
    uc = UtilityClass()

    # load docs 
    pdf_dir_path = '../data'
    response = uc.load_pdf_docs(pdf_dir_path)
    
    # split docs 
    split_docs_response = uc.split_docs_into_chunk(response['docs'])

    # create embedding models
    embedding_model = uc.load_embedding_model()

    # push to vectorstore 
    retriever = uc.push_to_vector_store(split_docs_response['chunks'], embedding_model, 10, 'Chroma')

    # do search
    query = "According to Kelly and Williams what is ethics?"
    search_response = uc.get_query_response(retriever, query)

    print(search_response)
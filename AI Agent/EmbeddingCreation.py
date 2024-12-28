import os
import gzip
import json
from langchain_ollama import OllamaEmbeddings                                                                                                   # type: ignore
from langchain_core.document import Document                                                                                                   # type: ignore
from langchain_core.text_splitters import RecursiveCharacterTextSplitter                                                                                                   # type: ignore
from langchain_chroma import Chroma                                                                                                   # type: ignore

def create_embeddings(wikipedia_filepath='./data/simplewiki-2020-11-01.jsonl.gz'):
    # Section: Embedding Model Initialization
    ollama_embed_model = OllamaEmbeddings(model='text-embedding-3-small')

    # Section: Document Loading and Preprocessing
    docs = []
    with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            data = json.loads(line.strip())
            docs.append({
                'metadata': {
                    'title': data.get('title'),
                    'article_id': data.get('id')
                },
                'data': ' '.join(data.get('paragraphs')[0:3])
            })

    docs = [doc for doc in docs for x in ['india'] if x in doc['data'].lower().split()]
    docs = [Document(page_content=doc['data'], metadata=doc['metadata']) for doc in docs]

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunked_docs = splitter.split_documents(docs)

    # Section: Chroma Database Initialization
    chroma_db = Chroma.from_documents(documents=chunked_docs,
                                      collection_name='rag_wikipedia_db',
                                      embedding=ollama_embed_model,
                                      collection_metadata={"hnsw:space": "cosine"},
                                      persist_directory="./wikipedia_db")
    
    return chroma_db

if __name__ == '__main__':
    create_embeddings()

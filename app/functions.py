from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from chromadb.config import Settings


import os
import tempfile
import uuid
import pandas as pd
import re


def clean_filename(filename):
    """Cleans up the filename by removing specific patterns."""
    return re.sub(r'\s\(\d+\)', '', filename)


def get_pdf_text(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        input_file = uploaded_file.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        return documents
    finally:
        os.unlink(temp_file.name)


def split_document(documents, chunk_size=1000, chunk_overlap=200):
    """Splits documents into chunks for better embedding performance."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " "]
    )
    return text_splitter.split_documents(documents)


def get_embedding_function(api_key):
    """Returns the embedding function using OpenAI."""
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key
    )


def create_vectorstore(chunks, embedding_function, file_name):
    """Creates a Chroma vectorstore explicitly in in-memory mode."""
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    unique_ids = set()
    unique_chunks = []

    for chunk, id in zip(chunks, ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)

    # Explicitly set persist_directory=None to force in-memory mode
    vectorstore = Chroma.from_documents(
        documents=unique_chunks,
        collection_name=clean_filename(file_name),
        embedding=embedding_function,
        ids=list(unique_ids),
        client_settings=Settings(
            persist_directory=None,  # This avoids using SQLite entirely
            anonymized_telemetry=False  # Optional: Disable telemetry
        )
    )
    return vectorstore


def create_vectorstore_from_texts(documents, api_key, file_name):
    """Processes documents and creates a Chroma vectorstore."""
    chunks = split_document(documents)
    embedding_function = get_embedding_function(api_key)
    vectorstore = create_vectorstore(chunks, embedding_function, file_name)
    return vectorstore


def format_docs(docs):
    """Formats documents into plain text."""
    return "\n\n".join(doc.page_content for doc in docs)


PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""


class AnswerWithSources(BaseModel):
    """Structure for answer, sources, and reasoning."""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")


class ExtractedInfoWithSources(BaseModel):
    """Extracted information structure for research articles."""
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources


def query_document(vectorstore, query, api_key):
    """Handles retrieval-augmented generation (RAG) for querying documents."""
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    retriever = vectorstore.as_retriever(search_type="similarity")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
    )

    structured_response = rag_chain.invoke(query)
    df = pd.DataFrame([structured_response.dict()])

    # Transforming into a table with rows for answers, sources, and reasoning
    rows = {key: [] for key in ['answer', 'source', 'reasoning']}
    for col in df.columns:
        rows['answer'].append(df[col][0]['answer'])
        rows['source'].append(df[col][0]['sources'])
        rows['reasoning'].append(df[col][0]['reasoning'])

    structured_response_df = pd.DataFrame(rows, index=df.columns)
    return structured_response_df.T

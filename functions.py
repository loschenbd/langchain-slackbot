from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def load_handbook():
    loader = PyPDFLoader("./data/Artisan_-_Employee_Handbook_-_May_2022_-_Signed.pdf")
    handbook_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(handbook_data)
    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(docs, embeddings)
    print(db)
    print(type(db))
    print(docs[0].page_content)
    return db


def get_response_from_query(db, query, k=4):
    print(db)
    print(query)
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query)
    docs_page_content = " ".join([d.page_content for d in docs])
    print(docs[0].page_content)
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that can answer questions about the Artisan Handbook 
        based on the data provided: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = f"Answer the following question: {query}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(
        llm=chat,
        prompt=chat_prompt,
        verbose=True
    )

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response


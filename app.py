from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.schema import Document
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["HF_API_KEY"]=os.getenv("HF_API_KEY")

arxiv_wrapper = ArxivAPIWrapper(top_k_search=1, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper = WikipediaAPIWrapper(top_k_search=1, doc_content_chars_max=500)
wikipedia = WikipediaQueryRun(api_wrapper=wiki_wrapper)
duckduckgo = DuckDuckGoSearchRun(name="Search")


def create_documents(docs):
    """Create documents from uploaded files."""
    
    loaders = {
        "txt": TextLoader,
        "pdf": PyMuPDFLoader,
        "docx": UnstructuredWordDocumentLoader,
    }

    documents = []
    for doc in docs:
        os.makedirs("./temp", exist_ok=True)
        tempdocs = os.path.join("./temp", doc.name)
        with open(tempdocs, "wb") as f: 
            f.write(doc.getbuffer())
        file_extension = f.name.split(".")[-1]
        if file_extension in loaders:
            loader = loaders[file_extension](f"./temp/" + doc.name)
            loaded_docs = loader.load()
            for loaded_doc in loaded_docs:
                # Extract page_content and metadata properly
                page_content = loaded_doc.page_content if hasattr(loaded_doc, "page_content") else ""
                metadata = loaded_doc.metadata if hasattr(loaded_doc, "metadata") else {}
                documents.append(Document(
                    page_content=page_content,
                    metadata=metadata
                ))
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    return docs

def create_embeddings(docs):
    """Create embeddings for the documents."""
    # Ensure docs have valid page_content
    for doc in docs:
        if not isinstance(doc.page_content, str):
            raise ValueError(f"Invalid page_content: {doc.page_content}")

    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    return vectorstore

st.set_page_config(
    page_title="Search Engine with Tools and Agents",
    page_icon="🤖",
    layout="centered",
)
st.title("Search Engine with Tools and Agents")
st.sidebar.title("settings")

api_key=st.sidebar.text_input("Enter GROQ key:", type="password")

if api_key:

    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    
    #create docs
    docs=create_documents(uploaded_file)
        
    # Create embeddings
    vectorstore = create_embeddings(docs)
    
    retriver=vectorstore.as_retriever()
    llm=ChatGroq(api_key=api_key,model_name="gemma2-9b-it", streaming=True)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriver)

    # Use the RetrievalQA Chain as a Tool
    retriver_tool = Tool(
        name="My Vector Store",
        func=qa_chain.run,
        description="Useful for answering questions based on the vector store."
    )
        
    #retriver_tool=create_retriever_tool(retriever=retriver, name="Vectorstore Retriever", description="Retrieves documents from the vectorstore.")
    #tools.append(retriver_tool)
    tools = [arxiv, wikipedia, duckduckgo, retriver_tool]
    
    #st.session_state.messages = []
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm a Chat Bot who can search the Web. How can I help you?"},
        ]
        
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt:=st.chat_input(placeholder="Ask me anything!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
       
        print(tools)
        agent = initialize_agent(
            tools,
            llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response=agent.run(st.session_state.messages, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
else:
    st.sidebar.warning("Please enter your API key to use the app.")

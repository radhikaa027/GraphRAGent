# agent.py (v6 - The Final, Polished Version)

# --- 1. IMPORTS & SETUP ---
import os
import fitz
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

load_dotenv()
print("Loaded environment variables.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
print("LLM initialized.")

uri = "bolt://localhost:7687"
user = "neo4j"
password = "company-data" # Your actual password
driver = GraphDatabase.driver(uri, auth=(user, password))
try:
    driver.verify_connectivity()
    print("Successfully connected to Neo4j!")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")

pdf_file_name = "Microsoft-10K.pdf"

# --- 2. BUILD OR LOAD PERSISTENT VECTOR STORE ---
persist_directory = "chroma_db_with_citations"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if os.path.exists(persist_directory):
    print(f"\nLoading existing vector store from: {persist_directory}")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Vector store loaded successfully.")
else:
    print(f"\nCreating new vector store and saving to: {persist_directory}")
    all_chunks, all_metadatas = [], []
    doc = fitz.open(pdf_file_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        chunks = text_splitter.split_text(page_text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"page": page_num + 1})
    print(f"Split document into {len(all_chunks)} chunks.")
    vector_store = Chroma.from_texts(
        texts=all_chunks, embedding=embeddings, metadatas=all_metadatas,
        persist_directory=persist_directory
    )
    print("New vector store created and saved successfully.")

retriever = vector_store.as_retriever()
print("Retriever created.")

# --- 3. THE DEFINITIVE TOOL DEFINITIONS ---

@tool
def semantic_search_tool(query: str) -> str:
    """
    Use this tool for general, open-ended questions about opinions, strategies, or summaries.
    This tool returns rich text snippets from the document. The output will include source citations
    like [Source, Page X], which YOU MUST INCLUDE in your final answer.
    """
    print(f"\n---> Using Semantic Search Tool for query: '{query}'")
    docs = retriever.invoke(query)
    return "\n\n".join(
        f"[Source, Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}"
        for doc in docs
    )

@tool
def graph_query_tool(query: str) -> str:
    """
    Use this tool for specific, factual questions about relationships between entities
    (e.g., 'Who acquired company X?', 'Who is the CEO of company Y?').
    This tool provides direct, factual answers from a knowledge graph and does NOT have page number citations.
    The answer from this tool is a primary fact and should be stated directly.
    """
    print(f"\n---> Using Graph Query Tool for query: '{query}'")
    cypher_generation_prompt = f"""
    You are a Neo4j expert. Given a question, generate a Cypher query to answer it.
    DATABASE SCHEMA: All nodes are labeled `:Entity` and have a `name` property.
    Question: {query}
    Cypher Query:
    """
    cypher_query = llm.invoke(cypher_generation_prompt).content.strip().replace("```cypher", "").replace("```", "")
    print(f"    Generated Cypher query: {cypher_query}")
    with driver.session() as session:
        result = session.run(cypher_query)
        return str(result.data())

tools = [semantic_search_tool, graph_query_tool]

# --- 4. THE DEFINITIVE AGENT AND PROMPT ---
# This prompt is simpler and more effective, relying on the tool descriptions for behavior.
system_instruction = """
You are a helpful assistant that answers questions based on the provided context.
Your goal is to provide a comprehensive and accurate answer.

When the context includes source citations like [Source, Page X], you MUST include these
in your final answer in the format [Page X].
"""

original_prompt = hub.pull("hwchase17/react")
new_template = system_instruction + "\n\n" + original_prompt.template
agent_prompt = PromptTemplate.from_template(new_template)

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print("\n--- Final Intelligent Agent is ready. Let's ask some questions! ---")

if __name__ == "__main__":
    # Question 1: Should be comprehensive AND have citations.
    question1 = "What did the report say about the company's strategy and future plans regarding AI?"
    response1 = agent_executor.invoke({"input": question1})
    print("\n--- FINAL ANSWER 1 ---")
    print(response1["output"])
    print("="*80)

    # Question 2: Should be direct and have NO citations.
    question2 = "What companies did Microsoft acquire?"
    response2 = agent_executor.invoke({"input": question2})
    print("\n--- FINAL ANSWER 2 ---")
    print(response2["output"])
    print("="*80)
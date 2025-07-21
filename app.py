import os
import shutil
import gradio as gr
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain import hub

# --- 1. SETUP ---
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
NEO4J_USERNAME = "neo4j"

# THIS IS THE CRITICAL DOCKER NETWORKING FIX
# If the cloud URI is not found, we are running locally in Docker.
# We must use the special Docker address to connect to the host PC.
if not NEO4J_URI:
    print("Cloud secrets not found, falling back to local Docker setup.")
    NEO4J_URI = "bolt://host.docker.internal:7687"
    NEO4J_PASSWORD = "company-data" # Your local password

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

readonly_db_path = "chroma_db_with_citations"
writable_db_path = "/data/chroma_db_persistent"
if os.path.exists("/data"):
    if not os.path.exists(writable_db_path):
        shutil.copytree(readonly_db_path, writable_db_path)
    persist_directory = writable_db_path
else:
    persist_directory = readonly_db_path

vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vector_store.as_retriever()
print("--- System Initialized Successfully ---")

# --- 2. TOOLS ---
@tool
def semantic_search_tool(query: str) -> str:
    """
    Use this tool for general, open-ended questions about opinions, strategies, or summaries.
    This tool returns rich text snippets from the document. The output will include source citations
    like [Source, Page X], which YOU MUST INCLUDE in your final answer.
    """
    print(f"\n---> Using Semantic Search Tool for query: '{query}'")
    docs = retriever.invoke(query)
    return "\n\n".join(f"[Source, Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}" for doc in docs)

@tool
def graph_query_tool(query: str) -> str:
    """
    Use this tool for specific, factual questions about relationships between entities
    (e.g., 'Who acquired company X?', 'Who is the CEO of company Y?').
    This tool provides direct, factual answers from a knowledge graph and does NOT have page number citations.
    The answer from this tool is a primary fact and should be stated directly.
    """
    print(f"\n---> Using Graph Query Tool for query: '{query}'")
    global generated_cypher_query # Use global variable to display in UI

    # THE DEFINITIVE, BULLETPROOF PROMPT
    cypher_generation_prompt = f"""
    You are an expert Neo4j Cypher query translator. Your sole purpose is to convert a
    user's question into a valid Cypher query.

    DATABASE SCHEMA: All nodes are labeled :Entity and have a name property (string).
    CRITICAL INSTRUCTIONS:
    1. Translate the user's question into a Cypher query.
    2. YOU MUST ONLY RETURN THE CYPHER QUERY. NOTHING ELSE.
    3. Do not add any explanation, preamble, or markdown formatting.
    4. If you cannot generate a query, return a simple query that will yield no results, like MATCH (n) WHERE false RETURN n.
    Question: {query}
    Cypher Query:"""
    generated_cypher_query = llm.invoke(cypher_generation_prompt).content.strip().replace("cypher", "").replace("", "")
    
    if not generated_cypher_query.strip().upper().startswith(('MATCH', 'RETURN')):
        print(f"LLM failed to generate a valid query. Returning empty.")
        return "[]"

    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run(generated_cypher_query)
            return str(result.data())
    except Exception as e:
        print(f"Error executing Cypher query: {e}")
        return "[]"
    finally:
        if driver:
            driver.close()

tools = [semantic_search_tool, graph_query_tool]
generated_cypher_query = ""

# --- 3. AGENT DEFINITION ---
# THE DEFINITIVE, NUANCED SYSTEM PROMPT
system_instruction = """
You are a specialized assistant for corporate intelligence. Your goal is to answer questions accurately based on the tools you have.
You have two tools available:
1. semantic_search_tool: Use this for general questions. It returns text from a document along with page number citations.
2. graph_query_tool: Use this for specific, factual questions about relationships. It returns structured data from a knowledge graph.
*YOUR BEHAVIOR:*
- When you use the semantic_search_tool, you MUST include source citations in the format [Page X].
- When you use the graph_query_tool, the result is a direct fact. This answer does NOT require a page number citation.
- If the graph_query_tool returns an empty result [], conclude that the fact is not in the database.
- Choose the best tool for the job.
"""

original_prompt = hub.pull("hwchase17/react")
new_template = system_instruction + "\n\n" + original_prompt.template
agent_prompt = PromptTemplate.from_template(new_template)
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)

# --- 4. GRADIO APPLICATION LOGIC ---
def agent_chat(message, history):
    history.append({"role": "user", "content": message})
    response = agent_executor.invoke({"input": message})
    final_answer = response['output']
    history.append({"role": "assistant", "content": final_answer})
    
    tool_name, thought_process, cypher_query, retrieved_context = "N/A", "N/A", "N/A", "N/A"
    
    if 'intermediate_steps' in response and response['intermediate_steps']:
        latest_action = response['intermediate_steps'][0][0]
        tool_name = latest_action.tool
        thought_process = latest_action.log
        if tool_name == "semantic_search_tool":
            retrieved_context = response['intermediate_steps'][0][1]
        elif tool_name == "graph_query_tool":
            global generated_cypher_query
            cypher_query = generated_cypher_query
            
    return history, thought_process, tool_name, cypher_query, retrieved_context

# --- 5. GRADIO UI DEFINITION ---
with gr.Blocks(theme=gr.themes.Soft(), title="Corporate Intelligence Agent") as demo:
    gr.Markdown("# Top-Notch Corporate Intelligence Agent 📈🤖")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", type="messages", height=500)
            msg = gr.Textbox(label="Your Question", placeholder="e.g., What is Microsoft's mission?")
            clear = gr.ClearButton([msg, chatbot])
        with gr.Column(scale=1):
            gr.Markdown("## Agent's Work 🕵")
            tool_display = gr.Textbox(label="Tool Used", interactive=False)
            thought_display = gr.Textbox(label="Agent's Thought Process", lines=10, interactive=False)
            with gr.Accordion("Retrieved Context / Cypher Query", open=False):
                cypher_display = gr.Code(label="Generated Cypher Query", interactive=False)
                context_display = gr.Textbox(label="Retrieved Semantic Context", lines=10, interactive=False)

    msg.submit(agent_chat, [msg, chatbot], [chatbot, thought_display, tool_display, cypher_display, context_display])

# --- THIS IS THE FINAL, CRITICAL FIX ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
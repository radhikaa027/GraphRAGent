# --- 1. IMPORTS & SETUP ---
import os
import fitz 
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv() 
print("Loaded environment variables.")

# Connect to the running Neo4j database
uri = "bolt://localhost:7687"
user = "neo4j"
password = "company-data" # Your actual password
driver = GraphDatabase.driver(uri, auth=(user, password))

# Check the connection to make sure it's working
try:
    driver.verify_connectivity()
    print("Successfully connected to Neo4j!")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
print("LLM initialized.")


# --- 2. DEFINE THE DATA STRUCTURE WE WANT (THE "SCHEMA") ---
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class Relationship(BaseModel):
    """A single relationship between two entities."""
    source_entity: str = Field(description="The name of the source entity (e.g., a person or company).")
    relationship_type: str = Field(description="The type of relationship (e.g., 'IS CEO OF', 'ACQUIRED', 'PARTNERED WITH').")
    target_entity: str = Field(description="The name of the target entity (e.g., a person or company).")

class GraphData(BaseModel):
    """A list of all relationships found in the text."""
    relationships: list[Relationship]


# --- 3. CREATE THE EXTRACTION LOGIC ---
parser = JsonOutputParser(pydantic_object=GraphData)
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
    You are an expert in analyzing corporate documents.
    From the text provided below, extract the key relationships between entities
    (people, companies, products, or technologies).

    Your goal is to create a knowledge graph. Focus on clear, direct relationships.
    For example:
    - "Satya Nadella is the CEO of Microsoft" -> (source: "Satya Nadella", type: "IS CEO OF", target: "Microsoft")
    - "Microsoft acquired GitHub" -> (source: "Microsoft", type: "ACQUIRED", target: "GitHub")

    {format_instructions}
    ---
    TEXT_TO_ANALYZE:
    {text_chunk}
    """,
    input_variables=["text_chunk"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

extraction_chain = prompt | llm | parser

print("Extraction chain created.")


# --- 4. FUNCTION TO ADD THE EXTRACTED DATA TO NEO4J ---
def add_to_graph(tx, graph_data):
    """
    This function is a "transaction" that takes the output of our AI chain
    and writes it into the Neo4j database. This version uses attribute access (e.g., rel.source_entity)
    which is safer than dictionary access (e.g., rel['source_entity']).
    """
    for rel in graph_data.relationships: # Accessing the list via attribute
        tx.run(
            """
            MERGE (a:Entity {{name: $source}})
            MERGE (b:Entity {{name: $target}})
            MERGE (a)-[r:{type}]->(b)
            """.format(type=rel.relationship_type.upper().replace(" ", "_")), 
            # These are the parameters for the Cypher query itself
            source=rel.source_entity,
            target=rel.target_entity
        )
        print(f"  Added to graph: ({rel.source_entity})-[:{rel.relationship_type.upper()}]->({rel.target_entity})")

# --- 5. MAIN SCRIPT EXECUTION ---
def ingest_pdf(pdf_path: str):
    """
    The main function that orchestrates the entire process.
    """
    print(f"\n--- Starting Ingestion for: {pdf_path} ---")
    doc = fitz.open(pdf_path)

    print("Reading text from the first 10 pages...")
    text_to_process = ""
    for page_num in range(min(137, doc.page_count)):
        text_to_process += doc[page_num].get_text()

    print("\nExtracting relationships with the LLM. This may take a moment...")
    extracted_data = extraction_chain.invoke({"text_chunk": text_to_process})
    
    print("\n--- DATA RECEIVED FROM LLM ---")
    import json
    print(json.dumps(extracted_data, indent=2))
    print("--- END OF DATA ---")
    
    
    # Convert the dictionary from the LLM into our Pydantic GraphData object.
    # The **extracted_data unpacks the dictionary into keyword arguments.
    graph_data_object = GraphData(**extracted_data)
    
    print("\nAdding extracted data to Neo4j graph...")
    # Now, we open a session to our database and run the transaction function.
    with driver.session() as session:
        # Pass the new Pydantic object to the function, not the raw dictionary
        session.execute_write(add_to_graph, graph_data_object)

    
    driver.close()
    print("\n--- Ingestion Complete ---")


if __name__ == "_main_":
    pdf_file_name = "Microsoft-10K.pdf" 
    ingest_pdf(pdf_file_name)
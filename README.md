# Graph-Enhanced & Agentic RAG for Corporate Intelligence 📈🤖

### A Project for Nebula9.ai

This project is a Question & Answering system designed to provide deep insights from complex business documents, such as corporate 10-K SEC filings. I have used Microsoft's 10K filing. 

The core innovation is its **hybrid RAG (Retrieval-Augmented Generation)** approach. It goes beyond simple semantic search by creating and querying a knowledge graph to understand the intricate relationships between entities mentioned in the text. This allows the system to answer not just "what" was said, but also "who" is connected to "what."

---

## 🚀 Live Demo

The application is fully containerized and deployed on Hugging Face Spaces. You can interact with the live version here without any setup required:

**➡️ [https://huggingface.co/spaces/Radhikaaaa/GraphRAGent-Project](https://huggingface.co/spaces/Radhikaaaa/GraphRAGent-Project) ⬅️**

> *(Note: The free tier Space may go to sleep after a period of inactivity. The first load may take up to 60 seconds to wake the container.)*
---

## 📹 Demo Video

Please find a short video demonstrating the project's full functionality here:

**➡️ [https://drive.google.com/file/d/1XA8Mk85Q1hJv-CecHXAV-JsvMEIVPgX3/view?usp=drive_link](https://drive.google.com/file/d/1XA8Mk85Q1hJv-CecHXAV-JsvMEIVPgX3/view?usp=drive_link) ⬅️**

---

## ✨ Key Features

- **Intelligent Agentic Router:** Uses a LangChain agent to analyze user queries and route them intelligently to the appropriate backend (graph or vector).
- **Dual Database Architecture:**
  - **Vector Database (ChromaDB):** For storing unstructured text chunks for semantic search.
  - **Graph Database (Neo4j):** For structured entities and relationships, enabling knowledge graph queries.
- **Source-Cited Answers:** Each answer includes document references with page numbers.
- **Persistent Vector Store:** ChromaDB stores embeddings on disk for faster startup.
- **Fully Containerized:** Dockerized for portability and reproducibility.

---

## 📊 Data Source

The primary data source for this project is Microsoft Corporation's annual 10-K filing for the fiscal year ended June 30, 2024. This document was sourced directly from the U.S. Securities and Exchange Commission's (SEC) EDGAR database.

*   **Filing:** Form 10-K (Annual Report)
*   **Company:** Microsoft Corporation (MSFT)
*   **Direct Link to Filing:** [SEC EDGAR Archives](https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm)

This document was chosen for its rich, unstructured text, containing numerous entities (executives, products, companies) and relationships (acquisitions, partnerships) that are ideal for demonstrating the capabilities of a hybrid Graph and Vector RAG system.

For convenience and reproducibility, the processed PDF version of this filing (`Microsoft-10K.pdf`) is included in this repository.

---

## 🏛️ Architecture

The system has two main pipelines:

### 1. Ingestion Pipeline (`ingest.py`)
- Loads a PDF document (Microsoft's 10-K).
- Uses an LLM (Google Gemini) to extract entities and relationships.
- Converts text chunks to vector embeddings and stores them in ChromaDB.
- Builds a Neo4j graph from extracted entities and links.

### 2. Query Pipeline (`app.py`)
- User submits a question via Gradio interface.
- LangChain agent decides:
  - If factual/relational → Queries Neo4j using Cypher.
  - If semantic/conceptual → Retrieves from ChromaDB.
- Synthesizes a final answer with citations.

---

## 🛠️ Tech Stack

- **Language:** Python
- **AI Framework:** LangChain
- **LLM:** Google Gemini 2.5 Flash
- **Databases:** Neo4j (Graph), ChromaDB (Vector)
- **Web UI:** Gradio
- **Containerization:** Docker

---

## 🧪 Getting Started (Local Setup)

### 1. Prerequisites

- [Git](https://git-scm.com/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Neo4j Desktop](https://neo4j.com/download/)

---

### 2. Setup Instructions

#### Step A: Clone the Repository
```bash
git clone https://github.com/radhikaa027/GraphRAGent.git
cd GraphRAGent 
```

#### Step B: Set Up Neo4j Database
    1. Open Neo4j Desktop and create a new local database.
    2. Set the password to company-data.
    3. Click the "Start" button to ensure the database is RUNNING.

#### Step C: Set Up Your API Key

    1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/).
    2. In the project's root directory, create a file named `.env`.
    3. Add the following line (without quotes): GOOGLE_API_KEY=YOUR_API_KEY_HERE

---

### 3. Build and Run the Application

#### Step A: Build the Docker Image
```bash
docker build -t corporate-agent .
```
#### Step B: Run the Ingestion Script (First Time Only)
```bash
docker run -it --rm --env-file .env --network="host" corporate-agent python ingest.py 
```

#### Step C: Launch the Gradio Web UI
```bash
docker run -p 7860:7860 -it --rm --env-file .env --network="host" corporate-agent 
```

---

### Step 4: Access the Application
Open your browser and go to:
```bash
http://localhost:7860
```
You can now interact with the agentic Q&A interface.


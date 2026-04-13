# DAO Proposal Intelligence Engine

AI and FastAPI service layer for a Decentralized Autonomous Organization (DAO) governance platform. The larger project is designed to help organizations such as universities, companies, and public-sector bodies submit proposals, analyze them with AI, and support transparent voting with verifiable records.

This repository focuses specifically on the AI layer: proposal intelligence, retrieval-augmented generation, bias-aware evaluation, proposal passing prediction, document summarization, and conversational querying over proposal content.

The system is built around Haystack-based RAG pipelines, Pinecone vector search, MongoDB-backed proposal data, and low-latency local sentence-transformer embeddings instead of external embedding APIs.

## Project Scope

This repository contains the AI layer of the larger DAO platform:

- AI / FastAPI repo: this repository
- Frontend/Backend repo: `https://github.com/Devansh-Aage/Arbiter`


## Broader Project Vision

The full DAO platform is intended to support:

- Proposal submission for policies, funding requests, curriculum changes, governance actions, and similar organizational decisions
- AI-assisted review using NLP, retrieval, summarization, outcome prediction, and bias-aware stakeholder analysis
- Secure voting with wallet-based authentication and gasless signature flows
- Transparent and auditable governance records, with optional blockchain-backed validation
- Flexible deployment across education, corporate, and government use cases

## What This Service Does

- Ingests proposal PDFs and converts them into searchable vector embeddings
- Uses Haystack pipelines to retrieve relevant context and generate AI responses
- Evaluates proposal acceptance likelihood using organization context
- Generates member-specific bias-aware voting summaries in the background
- Supports chat-based querying over proposal documents with stored conversation history
- Produces structured proposal summaries with accept/reject reasoning
- Appends post-voting AI-generated governance context back into the organization record
- Serves as the AI decision-support engine for a broader DAO governance workflow

## Key Highlights

- Haystack-centric RAG pipeline for proposal intelligence workflows
- Low embedding latency using `sentence-transformers/all-MiniLM-L6-v2`
- No external embedding API dependency for vector generation
- Pinecone vector store with namespace-based proposal isolation
- Groq-hosted LLM inference using `llama-3.1-8b-instant`
- FastAPI endpoints for evaluation, bias analysis, chat, and organization context generation
- MongoDB persistence for organizations, proposals, choices, votes, and messages
- AI contribution aligned with a larger governance system that can later integrate wallet auth, off-chain signed voting, and blockchain auditability

## Architecture

### 1. Document Ingestion

- Proposal PDFs are parsed and split into chunks
- Chunks are embedded using Sentence Transformers
- Embeddings are written to Pinecone for semantic retrieval

### 2. Retrieval-Augmented Generation

- User query or evaluation context is embedded
- Relevant proposal chunks are retrieved from Pinecone
- A Haystack `PromptBuilder` constructs task-specific prompts
- A Groq-backed LLM generates the final response

### 3. AI Workflows

- Outcome prediction: estimates proposal acceptance chance using retrieved content plus organization context
- Bias evaluation: checks proposal alignment against member bias/preferences
- Chat evaluation: answers proposal questions using retrieved document context and prior conversation
- Proposal summarization: generates structured JSON summary with acceptance and rejection reasons
- Context update: creates a concise governance outcome summary after voting is completed

## Main AI Features

- Proposal passing prediction
- Proposal summarization with structured accept/reject reasoning
- Stakeholder-aware bias summaries
- RAG chat over proposal documents
- Organization context enhancement after proposal completion
- AI support layer that can be extended toward agentic governance and tool-augmented admin workflows

## Tech Stack

- Python
- FastAPI
- Streamlit
- Haystack
- Sentence Transformers
- Pinecone
- MongoDB with Motor
- Groq API for LLM generation
- PyMuPDF / PDF tooling

## Repository Structure

```text
app/
  app.py                  Streamlit interface for local testing/demo
  server.py               FastAPI server and REST endpoints
  config.py               Model and infrastructure configuration
  core/
    ingest.py             PDF parsing, chunking, embedding, indexing
    pipeline.py           Unified Haystack RAG pipeline
    document_store.py     Pinecone document store setup
    summarizer.py         JSON proposal summarization
    context_update.py     Governance outcome context generation
  services/
    outcome_service.py    Proposal acceptance evaluation
    bias_service.py       Bias-aware proposal analysis
    bias_background.py    Async member-specific bias processing
    chat_service.py       Chat-based RAG service
    db_service.py         MongoDB data access layer
  prompts/
    chat.txt
    bias.txt
    outcome.txt
tools/
  history_data.py
```

## API Endpoints

### `GET /`

Health check.

### `GET /organizations`

Returns available organizations from MongoDB.

### `POST /evaluate`

Uploads a proposal PDF and organization ID, indexes the document, retrieves organization context, and returns an AI-generated proposal outcome evaluation.

### `POST /bias-evaluate`

Creates a proposal, generates a structured summary, stores proposal choices, indexes the proposal document, and triggers background member-level bias analysis.

### `POST /chat-evaluate`

Loads stored message history for a user and proposal, retrieves proposal context, and returns an AI-generated chat response.

### `POST /generate-org-context`

Generates a short governance outcome summary after voting and appends it to the organization context.

## Environment Variables

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key
MONGO_URI=your_mongodb_connection_string
```

## Important Configuration

Current defaults from `app/config.py`:

- Pinecone index: `updatedvectors`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM model: `llama-3.1-8b-instant`
- Chunk size: `300`
- Chunk overlap: `50`
- Vector dimension: `384`

## Local Setup

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
uvicorn app.server:app --reload
```

Run the Streamlit app:

```bash
streamlit run app/app.py
```
# CSV Agent

🤖 **Intelligent CSV Agent**

This project is an interactive web application built with Streamlit that allows users to upload CSV files (or ZIP archives containing CSVs), preview the data, and interact with an AI agent to ask questions about the file contents.

## Features
- Upload `.csv` files or `.zip` archives (with CSVs)
- Data table preview
- Chat with an AI agent to ask questions about the data
- Conversation history

## How to Use

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   Create a `.env` file in the project root with the following variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_BASE_URL=your_openai_base_url
   ```

3. **Run the app:**
   ```bash
   streamlit run main.py
   ```

4. **Open in your browser:**
   Streamlit will open a local page where you can upload files and interact with the agent.

## Project Structure
```
csv-agent/
│
├── agent.py
├── main.py
├── processor.py
├── requirements.txt
├── .env
├── data/
│   ├── 202401_NFs_Cabecalho.csv
│   ├── 202401_NFs_Itens.csv
│   └── 202401_NFs.zip
└── README.md
```

## Main Dependencies
- streamlit
- python-dotenv
- pandas
- llama-index
- langchain
- openai
- huggingface-hub

## Internal Architecture

The agent's internal workflow is structured as follows:

```
User asks a question in natural language
        │
        ▼
OpenAI GPT-3.5 Turbo (via OpenAI API) ←───────────────┐
        │                                             │
        ▼                                             │
Query Engine interprets the question                  │
        │                                             │
        ▼                                             │
Vector Index over the CSVs                            │
        │                                             │
        ▼                                             │
SimpleDirectoryReader loads the files                 │
        └─────────────────────────────────────────────┘
```

- **User** interacts via the Streamlit interface, asking questions about the CSV data.
- **SimpleDirectoryReader** loads all CSV files from the `data/` directory.
- **Vector Index** (LlamaIndex) creates an embedding-based index over the CSV contents.
- **Query Engine** (LlamaIndex + LangChain) interprets the user's question and formulates a query.
- **OpenAI GPT-3.5 Turbo** is called as the LLM to generate a natural language answer based on the indexed data.
- The answer is returned to the user in the chat interface.

This architecture allows the agent to efficiently search, interpret, and answer questions about any uploaded CSV file using advanced language models and vector search.

## Notes
- The agent uses OpenAI and HuggingFace models to answer questions about the uploaded data.
- Make sure the `data/` folder exists and is accessible for file upload and reading.

---

Developed by [Your Name].

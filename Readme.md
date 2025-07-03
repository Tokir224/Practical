# RAG-Practical

A practical implementation of Retrieval-Augmented Generation (RAG) system with chunking evaluation pipeline.

## Prerequisites

- Python 3.12

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Tokir224/Practical.git
cd Practical
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install FAISS (choose one based on your system)
# For CPU-only systems:
pip install faiss-cpu

# For systems with GPU support:
pip install faiss-gpu
```

### 4. Environment Configuration

Create a `.env` file in the project root directory with the following variables:

```env
GROQ_API_KEY=""
LANGSMITH_API_KEY=""
```

**Note:** Make sure to fill in your actual API keys between the quotes.

## Usage

### 1. Run Chunking Evaluation Pipeline

```bash
python chunking_evaluation_pipeline.py
```

### 2. Run Main Application

```bash
uvicorn main:app
#Note: If you are using the faiss-gpu library, make sure to install a compatible version of NumPy:
pip install "numpy<2.0"
```

## API Keys Setup

### GROQ API Key
1. Visit [Groq Console](https://console.groq.com/)
2. Create an account or sign in
3. Generate an API key
4. Add it to your `.env` file

### LangSmith API Key
1. Visit [LangSmith](https://smith.langchain.com/)
2. Create an account or sign in
3. Generate an API key
4. Add it to your `.env` file

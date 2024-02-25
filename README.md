# SimpleRAG

# Retrieval-Augmented Generation (RAG) Implementation

This repository contains an implementation of Retrieval-Augmented Generation (RAG) using Python, ChainLit, Hugging Face, and OLLAMA. RAG enhances the capabilities of generative language models by integrating retrieval-based methods to provide more contextually relevant responses.

## Overview

RAG combines retrieval-based techniques with generative language models to improve response quality by retrieving relevant information from a knowledge source before generating responses. This implementation leverages ChainLit for handling chains of inference, Hugging Face for pre-trained language models, and OLLAMA for accessing external knowledge sources.

## Requirements

- Python 3.x
- ChainLit (installation instructions [here](https://chainlit.org/))
- HuggingFace Transformers ([here](https://huggingface.co/transformers/installation.html))
- OLLAMA (installation instructions [here](https://ollama.com))

## Installation

1. Clone this repository and change directory:
```
https://github.com/EngineersInsights/SimpleRAG.git
cd SimpleRAG
```

2. Install dependencies
```
pipenv shell
pipenv install 
```
3. Ingest the data to create vectors:
```
python ingest.py
```

4. Run the model:
```
python dev_runner.py
```

#### ⚠️⚠️Make sure you have ollama installed and running with llama2 model :⚠️⚠️
OLLAMA (installation instructions [here](https://ollama.com))

### If something is not working DM us on instagram: 

### Engineers Insights ([engineers_insights](https://www.instagram.com/engineers_insights/))
### ENJOY :) 

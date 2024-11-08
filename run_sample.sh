#!/bin/bash

# Navigate to the specified directory
cd /data/llm/LLM-TPU || { echo "Directory not found"; exit 1; }

# Source the virtual environment
source /data/llm/llm_venv/bin/activate

# Run the gradio script
python run_gradio_chat.py

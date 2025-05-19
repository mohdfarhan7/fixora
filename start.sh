#!/bin/bash

# Create necessary directories
mkdir -p images
mkdir -p faiss_index

# Check if PDF exists
if [ ! -f "OMT312630 (3).pdf" ]; then
    echo "Error: PDF file not found!"
    exit 1
fi

# Run main.py to ensure index and images are up to date
echo "Running main.py to set up vector store and images..."
python3 main.py

# Check if FAISS index exists
if [ ! -d "faiss_index" ] || [ ! -f "faiss_index/index.faiss" ]; then
    echo "Error: FAISS index not found!"
    exit 1
fi

# Check if images directory exists and has content
if [ ! -d "images" ] || [ -z "$(ls -A images)" ]; then
    echo "Error: Images directory is empty or not found!"
    exit 1
fi

# Start the application
uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-10000} 
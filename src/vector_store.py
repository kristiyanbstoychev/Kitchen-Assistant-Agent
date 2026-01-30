"""
Vector Store Handler for Kitchen Inventory Agent

This module handles:
1. Loading text files from knowledge base
2. Converting text to embeddings using Ollama
3. Storing embeddings in ChromaDB
4. Querying for relevant information
"""

import os
import chromadb
from chromadb.config import Settings
import requests
import json


class VectorStore:
    """
    Manages the vector database for inventory knowledge.
    
    Key Concepts:
    - Collection: Like a database table, stores all our inventory embeddings
    - Embedding: Numerical representation of text (1024 numbers)
    - Similarity Search: Finding documents with similar meanings
    """
    
    def __init__(self, persist_directory="./vector_db", collection_name="inventory"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Where to save the database on disk
            collection_name: Name of the collection (like a table name)
        """
        # Create ChromaDB client that saves to disk
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False  # Don't send usage data
        ))
        
        # Get or create collection
        # Collections store embeddings + metadata + original text
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Restaurant kitchen inventory data"}
        )
        
        # Ollama API endpoint for embeddings
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.embedding_model = "mxbai-embed-large"
    
    def get_embedding(self, text):
        """
        Convert text into a vector using Ollama.
        
        Args:
            text: String to convert (e.g., "olive oil inventory")
            
        Returns:
            List of 1024 numbers representing the text's meaning
            
        How it works:
        1. Send text to Ollama API
        2. Ollama runs mxbai-embed-large model
        3. Returns 1024-dimensional vector
        """
        payload = {
            "model": self.embedding_model,
            "prompt": text
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def load_knowledge_base(self, knowledge_base_dir="./knowledge_base"):
        """
        Load all .txt files from knowledge base and vectorize them.
        
        Process:
        1. Read each .txt file
        2. Split into chunks (each file = 1 chunk for simplicity)
        3. Generate embedding for each chunk
        4. Store in ChromaDB with metadata
        
        Args:
            knowledge_base_dir: Path to folder with .txt files
        """
        # Check if collection already has data
        if self.collection.count() > 0:
            print(f"Collection already contains {self.collection.count()} documents.")
            print("Skipping loading. Delete vector_db folder to reload.")
            return
        
        print("Loading knowledge base...")
        
        # List all .txt files
        files = [f for f in os.listdir(knowledge_base_dir) if f.endswith('.txt')]
        
        for file_name in files:
            file_path = os.path.join(knowledge_base_dir, file_name)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate embedding
            embedding = self.get_embedding(content)
            
            if embedding:
                # Add to ChromaDB
                # - ids: Unique identifier for this document
                # - embeddings: The vector representation
                # - documents: Original text (stored for retrieval)
                # - metadatas: Extra info (file source)
                self.collection.add(
                    ids=[file_name],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{"source": file_name}]
                )
                print(f"✓ Loaded {file_name}")
        
        print(f"Knowledge base loaded: {self.collection.count()} documents")
    
    def search(self, query, n_results=2):
        """
        Find the most relevant documents for a query.
        
        Args:
            query: User's question (e.g., "how much olive oil?")
            n_results: How many relevant documents to return
            
        Returns:
            List of relevant text chunks
            
        How similarity search works:
        1. Convert query to embedding
        2. Compare query embedding to all stored embeddings
        3. Return closest matches (cosine similarity)
        
        Example:
        Query: "olive oil stock"
        → Finds oils_fats.txt because "olive oil" is in that document
        """
        # Convert query to vector
        query_embedding = self.get_embedding(query)
        
        if not query_embedding:
            return []
        
        # Search for similar vectors
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Extract the actual text documents
        if results and results['documents']:
            return results['documents'][0]  # Returns list of matching documents
        
        return []
    
    def get_all_inventory(self):
        """
        Retrieve all inventory documents.
        Useful for generating complete reports.
        
        Returns:
            List of all document contents
        """
        # Get all documents from collection
        all_docs = self.collection.get()
        return all_docs['documents'] if all_docs else []
import os

import chromadb

# Update this to point to the directory containing all documents
DOCUMENTS_DIRECTORY = "documents"

# Instantiate Chroma Client
client = chromadb.PersistentClient("vector_data/chroma")
collection = client.get_or_create_collection("test_collection")

for doc_num, file_name in enumerate(DOCUMENTS_DIRECTORY):
    file_path = os.path.join(DOCUMENTS_DIRECTORY, file_name)
    
    # Check if the item is a file and not a directory
    if os.path.isfile(file_path):
        # Open the file and read its contents
        with open(file_path, "r", encoding="utf-8") as file:
            document_text = file.read()
            # Add document to Vector DB
            collection.upsert(documents=[document_text], metadatas=[{"source": file_name}], ids=[f"id{doc_num+1}"])
            
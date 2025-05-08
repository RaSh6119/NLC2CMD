from chromadb import PersistentClient

# Step 1: Connect to ChromaDB
client = PersistentClient(path="../../utils/chroma_db_1")  # adjust the path if needed

# Step 2: Load the 'cmd_train' collection
collection = client.get_or_create_collection("cmd_train")

# Step 3: Fetch first 5 entries
results = collection.get(limit=5)

# Step 4: Print what's stored
for i in range(len(results["ids"])):
    print(f"\n🆔 ID: {results['ids'][i]}")
    print(f"📝 Invocation: {results['documents'][i]}")
    print(f"⚙️ Command (metadata): {results['metadatas'][i].get('cmd')}")

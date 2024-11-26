from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility
import mysql.connector

# Set up MySQL database connection details
db_config = {
    'user': 'root',
    'password': 'Buggyx23',
    'host': 'localhost',
    'database': 'nature_oncology_journals'
}

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Function to recreate Milvus collection
def recreate_milvus_collection():
    # Check if the collection exists
    if utility.has_collection("journal_titles"):
        utility.drop_collection("journal_titles")  # Drop the collection if it exists

    # Create a new collection
    fields = [
        FieldSchema(name="title_id", dtype=DataType.INT64, is_primary=True, auto_id=False),  # Use custom IDs
        FieldSchema(name="title_embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields, "Oncology journal title embeddings")
    collection = Collection("journal_titles", schema=schema)
    return collection

collection = recreate_milvus_collection()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieve journal titles from MySQL and store embeddings in Milvus
def embed_and_store_in_milvus():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("SELECT id, title FROM journals")  # Include ID in your selection

        titles = cursor.fetchall()
        embeddings = []
        ids = []  # List to hold the IDs

        for (id, title) in titles:
            # Generate embedding
            embedding = model.encode(title).tolist()
            embeddings.append(embedding)
            ids.append(id)  # Store the corresponding ID

            # Print for debugging
            print(f"Storing title: {title} with ID: {id}")

        # Prepare data for insertion
        # Data should be structured as a list of lists
        data_to_insert = [
            ids,         # List of IDs
            embeddings   # List of embeddings
        ]

        # Insert embeddings into Milvus with corresponding IDs
        collection.insert(data_to_insert)  # Pass the prepared data structure
        print("Embeddings stored in Milvus.")

    except mysql.connector.Error as err:
        print("Error:", err)
    finally:
        cursor.close()
        connection.close()

# Call the function to embed and store journal titles in Milvus
embed_and_store_in_milvus()

# Create index on title_embedding field for search
index_params = {
    "index_type": "IVF_FLAT",  # You can also use "IVF_SQ8" or "HNSW"
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index(field_name="title_embedding", index_params=index_params)

# Load collection into memory after creating the index
collection.load()

def search_articles(query, top_k=5):
    # Step 1: Convert the query to an embedding
    query_embedding = model.encode(query).tolist()
    print(f"Query embedding: {query_embedding[:5]}...")  # Debug: Print part of the embedding

    # Step 2: Perform the search in Milvus
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    results = collection.search(
        data=[query_embedding], 
        anns_field="title_embedding", 
        param=search_params, 
        limit=top_k, 
        output_fields=["title_id"]
    )

    print(f"Search results: {results}")  # Debug: Print search results

    # Step 3: Extract the `title_id` for the top results
    title_ids = [hit.id for hit in results[0]] if results[0] else []
    print(f"Title IDs: {title_ids}")  # Debug: Print title IDs

    # Step 4: Retrieve article details from MySQL based on `title_id`
    if title_ids:
        try:
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()

            format_strings = ','.join(['%s'] * len(title_ids))
            # Select the title and abstract columns instead of title and details
            cursor.execute(f"SELECT title, abstract FROM journals WHERE id IN ({format_strings})", tuple(title_ids))

            articles = cursor.fetchall()

            # Print the article details
            for article in articles:
                title, abstract = article  # Use abstract instead of details
                print(f"Title: {title}\nAbstract: {abstract}\n")

        except mysql.connector.Error as err:
            print("Error:", err)
        finally:
            cursor.close()
            connection.close()
    else:
        print("No matching articles found.")  # Debug: Handle case where no IDs are returned


# Example query
query = "Prostaglandin E2-EP2/EP4 signaling induces immunosuppression in human cancer by impairing bioenergetics and ribosome biogenesis in immune cells"
search_articles(query)

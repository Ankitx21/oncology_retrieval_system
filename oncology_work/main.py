# main.py
import requests
from bs4 import BeautifulSoup
import mysql.connector
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility
import configparser

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# MySQL Database Connection
db_config = {
    'user': config['mysql']['user'],
    'password': config['mysql']['password'],
    'host': config['mysql']['host'],
    'database': config['mysql']['database']
}

# Connect to Milvus
connections.connect("default", host=config['milvus']['host'], port=config['milvus']['port'])

def save_to_database(title, authors, published_date, abstract):
    """Saves article data to MySQL database, ensuring IDs are sequential.
    The function retrieves the current maximum ID to maintain sequential IDs
    when inserting new records into the 'journals' table.
    """
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Get the current maximum ID
        cursor.execute("SELECT MAX(id) FROM journals")
        max_id = cursor.fetchone()[0]
        new_id = 1 if max_id is None else max_id + 1  # Start from 1 if there are no records

        insert_query = """
        INSERT INTO journals (id, title, authors, published_date, abstract)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (new_id, title, authors, published_date, abstract))
        connection.commit()
        print("Article saved to database:", title)

    except mysql.connector.Error as err:
        print("Error:", err)
    finally:
        cursor.close()
        connection.close()


def get_article_details(article_url):
    """Retrieves article details and saves them to the database.
    This function fetches the article data from the provided URL, extracts
    the title, authors, published date, and abstract, and then saves this data 
    to the MySQL database using the 'save_to_database' function.
    """
    response = requests.get(article_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.find("h1", class_="c-article-title").get_text(strip=True)
    published_date_text = soup.find("time").get_text(strip=True)
    published_date = datetime.strptime(published_date_text, "%d %B %Y").date()
    author_list = [author.get_text(strip=True) for author in soup.select("ul.c-article-author-list a[data-test='author-name']")]
    authors = ", ".join(author_list)
    abstract = soup.find("div", class_="c-article-section__content").get_text(strip=True)

    save_to_database(title, authors, published_date, abstract)

def get_latest_research_urls(url):
    """Fetches the latest research URLs and retrieves article details.
    This function scrapes the specified URL to find the section for the latest 
    research and reviews, and then calls 'get_article_details' for each article link found.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    base_url = "https://www.nature.com"
    latest_research_section = soup.find("h2", class_="c-section-heading", string="Latest Research and Reviews")

    if latest_research_section:
        section_container = latest_research_section.find_next_sibling()
        if section_container:
            for item in section_container.select("h3.c-card__title a"):
                full_link = base_url + item['href']
                get_article_details(full_link)

def recreate_milvus_collection():
    """Recreates the Milvus collection for storing embeddings.
    This function checks if the collection 'journal_titles' exists; if it does, 
    it drops the collection and creates a new one with the specified schema for 
    storing article title embeddings.
    """
    if utility.has_collection("journal_titles"):
        utility.drop_collection("journal_titles")
    fields = [
        FieldSchema(name="title_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="title_embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields, "Oncology journal title embeddings")
    collection = Collection("journal_titles", schema=schema)
    return collection

def embed_and_store_in_milvus(collection):
    """Embeds journal titles and stores them in Milvus.
    This function retrieves journal titles from the MySQL database, generates 
    embeddings for each title using a pre-trained model, and stores the 
    embeddings along with their IDs in the specified Milvus collection.
    """
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("SELECT id, title FROM journals")
        titles = cursor.fetchall()
        embeddings = []
        ids = []

        model = SentenceTransformer("all-MiniLM-L6-v2")

        for (id, title) in titles:
            embedding = model.encode(title).tolist()
            embeddings.append(embedding)
            ids.append(id)
            print(f"Storing title: {title} with ID: {id}")

        data_to_insert = [ids, embeddings]
        collection.insert(data_to_insert)
        print("Embeddings stored in Milvus.")

    except mysql.connector.Error as err:
        print("Error:", err)
    finally:
        cursor.close()
        connection.close()

def create_index(collection):
    """Creates an index on the title_embedding field for search.
    This function defines the parameters for the indexing of the title embeddings 
    in the Milvus collection to enable efficient retrieval during search operations.
    """
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="title_embedding", index_params=index_params)
    collection.load()

def search_articles(query, collection, top_k=5):
    """Searches for articles in Milvus based on the query.
    This function converts the search query into an embedding, performs a 
    similarity search in the Milvus collection, and retrieves the top K articles 
    that match the query based on the embeddings.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()
    print(f"Query embedding: {query_embedding[:5]}...")

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

    title_ids = [hit.id for hit in results[0]] if results[0] else []
    print(f"Title IDs: {title_ids}")

    if title_ids:
        try:
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()
            format_strings = ','.join(['%s'] * len(title_ids))
            cursor.execute(f"SELECT title, abstract FROM journals WHERE id IN ({format_strings})", tuple(title_ids))
            articles = cursor.fetchall()

            for article in articles:
                title, abstract = article
                print(f"Title: {title}\nAbstract: {abstract}\n")

        except mysql.connector.Error as err:
            print("Error:", err)
        finally:
            cursor.close()
            connection.close()
    else:
        print("No matching articles found.")

# Main Execution
if __name__ == "__main__":
    # Fetch and store articles in the database
    url = "https://www.nature.com/subjects/oncology"
    get_latest_research_urls(url)

    # Create Milvus collection and embed journal titles
    collection = recreate_milvus_collection()
    embed_and_store_in_milvus(collection)
    create_index(collection)

    # Example query for searching articles
    query = "Prostaglandin E2-EP2/EP4 signaling induces immunosuppression in human cancer by impairing bioenergetics and ribosome biogenesis in immune cells"
    search_articles(query, collection)

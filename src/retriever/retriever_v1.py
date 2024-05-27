import os

from elasticsearch import Elasticsearch


def search(query, index="wikipedia", num_docs=5):
    """
    Search the Elasticsearch index for the most relevant documents.
    """

    docs = []
    if num_docs > 0:
        print(f"Running query: {query}")
        es_request_body = {
            "query": {
                "match": {"content": query}  # Assuming documents have a "content" field
            },
            "size": num_docs,
        }

        # Connect to Elasticsearch
        es = Elasticsearch(
            hosts=os.environ["ES_HOST"],
            basic_auth=("elastic", os.environ["ES_PASSWORD"]),
            verify_certs=False,
            ssl_show_warn=False,
        )

        response = es.options(request_timeout=60).search(
            index=index, body=es_request_body
        )
        # Extract and return the documents
        # docs = [f"Document {hit["_source"]["_id"]}\n{hit["_source"]["content"]}" for hit in response["hits"]["hits"]]
        print(response["hits"]["hits"])

        def hit_to_str(hit):
            res = hit["_source"]["content"]
            if "filename" in hit["_source"]:
                res = f"Document: { hit['_source']['filename']}\n{hit['_source']['content']}"
            return res

        docs = [hit_to_str(hit) for hit in response["hits"]["hits"]]
        print(f"Received {len(docs)} documents from index {index}")

    return docs

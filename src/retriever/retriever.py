import json
import os

os.environ["PYSERINI_CACHE"] = "/data/cache/"
os.environ["JAVA_TOOL_OPTIONS"] = "-Xms6400m -Xmx12800m"
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder


class Retriever:
    _instances = {}

    def __new__(cls, index_type):
        if index_type not in cls._instances:
            cls._instances[index_type] = super(Retriever, cls).__new__(cls)
            cls._instances[index_type]._init_searchers(index_type)
        return cls._instances[index_type]

    def _init_searchers(self, index_type):
        self.index_type = index_type

        if index_type == "tct_colbert-msmarco":
            encoder = TctColBertQueryEncoder("castorini/tct_colbert-msmarco")
            self.searcher = FaissSearcher.from_prebuilt_index(
                "msmarco-passage-tct_colbert-hnsw", encoder
            )
        elif index_type == "msmarco-v1-passage":
            self.searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
        elif index_type == "wikipedia-dpr":
            self.searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr")
        else:
            self.searcher = LuceneSearcher.from_prebuilt_index("enwiki-paragraphs")

    def retrieve_paragraphs(self, query, num_ret=5):
        hits = self.searcher.search(query, num_ret)
        paragraphs = []
        for i in range(len(hits)):
            doc = self.searcher.doc(hits[i].docid)
            if doc.raw()[0] == "{":
                json_doc = json.loads(doc.raw())
                para = json_doc["contents"]
            else:
                para = doc.raw()
            paragraphs.append(para)

        return paragraphs

import numpy as np
from pymilvus.orm import utility
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer


class MilvusVectorStore:
    def __init__(
        self,
        collection_name: str,
        model_path: str,
        host: str = "localhost",
        port: str = "19530",
        recreate: bool = False,
    ):
        self.collection_name = collection_name
        self.model_path = model_path
        self.dim = None
        self.collection = None
        self.local_mode = False
        self.local_embeddings = None
        self.model = SentenceTransformer(model_path)
        self._connect_milvus(host, port)

        if self.local_mode:
            self.collection_exists = False
            return

        self.collection_exists = utility.has_collection(collection_name)
        if recreate and self.collection_exists:
            utility.drop_collection(collection_name)
            self.collection_exists = False

    def _connect_milvus(self, host, port):
        try:
            connections.connect("default", host=host, port=port, timeout=3)
        except Exception as exc:
            self.local_mode = True
            print(f"[MilvusVectorStore] Milvus unavailable, using local vector search: {exc}")

    def create_collection(self, texts: list):
        if self.local_mode:
            self.local_embeddings = None
            self.collection_exists = True
            return

        if self.collection_exists:
            print(f"Collection {self.collection_name} already exists, loading it")
            self.load_collection()
            return

        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        self.dim = embeddings.shape[1]
        ids = list(range(len(embeddings)))

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields, description="Text embedding collection")

        self.collection = Collection(name=self.collection_name, schema=schema)
        self.collection.insert([ids, embeddings.tolist()])
        self.collection.flush()
        self.collection.create_index(
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
        )
        self.collection.load()
        self.collection_exists = True
        print(f"Created collection: {self.collection_name}")

    def load_collection(self):
        if self.local_mode:
            return
        self.collection = Collection(name=self.collection_name)
        self.collection.load()

    def get_collection(self):
        return self.collection

    def search(self, query_text: str, corpus: list, top_k: int = 15):
        prompted_query = "为这个句子生成表示以用于检索相关文章：" + query_text
        query_vec = self.model.encode([prompted_query], convert_to_numpy=True).tolist()

        if self.local_mode:
            if self.local_embeddings is None:
                return [{"content": "", "score": 0.0, "source": "localvectorsearch"}]
            query_arr = np.asarray(query_vec[0], dtype=np.float32)
            doc_arr = np.asarray(self.local_embeddings, dtype=np.float32)
            query_norm = np.linalg.norm(query_arr) or 1.0
            doc_norms = np.linalg.norm(doc_arr, axis=1)
            doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)
            scores = doc_arr @ query_arr / (doc_norms * query_norm)
            order = np.argsort(scores)[::-1][:top_k]
            return [
                {
                    "content": corpus[int(idx)],
                    "score": round(float(scores[int(idx)]), 3),
                    "source": "localvectorsearch",
                }
                for idx in order
            ]

        results = self.collection.search(
            data=query_vec,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 8}},
            limit=top_k,
            output_fields=["id"],
        )

        output = []
        for result in results[0]:
            idx = int(result.entity.get("id"))
            output.append(
                {
                    "content": corpus[idx],
                    "score": round(result.score, 3),
                    "source": "vectorsearch",
                }
            )
        return output

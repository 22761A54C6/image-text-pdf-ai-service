"""
DB Injection Pipeline
======================
One orchestrated flow: fetch -> clean (optional Groq) -> embed (Voyage AI)
-> inject (Mongo upsert) -> ensure vector index.

Replaces the old "wipe collection, loop, insert one-by-one" pattern in
sync_categories.py with a single reusable class any data source can use.

Usage:
    from app.db_injection_pipeline import DBInjectionPipeline

    pipeline = DBInjectionPipeline(
        collection_name="categories",
        text_field="name",
        id_field="sourceId",
        vector_index_name="category_vector_index",  # matches matching.py
    )
    pipeline.run(source=categories_list)
"""
from typing import List, Union

import json
import requests
from pymongo import UpdateOne

from app.database import db
from app.embeddings import get_embeddings_batch
from app.config import VOYAGE_EMBED_DIMENSIONS, GROQ_API_KEY, GROQ_URL, GROQ_MODEL


def normalize_record_to_text(record: dict) -> str:
    """Turns one raw JSON record into a single clean natural-language
    description suitable for embedding, via Groq. Falls back to a naive
    join of the record's values if Groq fails, so a bad LLM call never
    blocks the load. Only used when use_llm_cleanup=True below -- skip it
    for clean data (like a plain categories list) to avoid an LLM call
    on data that doesn't need cleaning."""
    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Convert the given JSON record into one concise, "
                            "natural-language sentence describing it, suitable "
                            "for semantic search embedding. Output ONLY the "
                            "sentence -- no preamble, no quotes, no markdown."
                        ),
                    },
                    {"role": "user", "content": json.dumps(record)},
                ],
                "temperature": 0,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[llm_preprocess] Groq normalization failed, falling back to raw join: {e}")
        return " ".join(str(v) for v in record.values() if v)


class DBInjectionPipeline:
    def __init__(
        self,
        collection_name: str,
        text_field: str = "name",
        id_field: str = None,
        use_llm_cleanup: bool = False,
        vector_index_name: str = "vector_index",
        num_dimensions: int = VOYAGE_EMBED_DIMENSIONS,
    ):
        self.collection_name = collection_name
        self.text_field = text_field
        self.id_field = id_field
        self.use_llm_cleanup = use_llm_cleanup
        self.vector_index_name = vector_index_name
        self.num_dimensions = num_dimensions
        self.collection = db[collection_name]

    def _fetch(self, source: Union[str, list]) -> List[dict]:
        if isinstance(source, list):
            records = source
        else:
            import json
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
        print(f"[pipeline:fetch] {len(records)} records from source")
        return records

    def _clean(self, records: List[dict]) -> List[str]:
        if self.use_llm_cleanup:
            texts = [normalize_record_to_text(r) for r in records]
            print(f"[pipeline:clean] {len(texts)} records normalized via Groq")
        else:
            texts = [str(r.get(self.text_field, "")) for r in records]
            print(f"[pipeline:clean] {len(texts)} records using raw '{self.text_field}' field")
        return texts

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = get_embeddings_batch(texts, input_type="document")
        print(f"[pipeline:embed] {len(embeddings)} vectors generated via Voyage AI")
        return embeddings

    def _inject(self, records, texts, embeddings) -> int:
        if self.id_field:
            operations = []
            for record, text, embedding in zip(records, texts, embeddings):
                doc = dict(record)
                doc["embeddedText"] = text
                doc["embedding"] = embedding
                operations.append(
                    UpdateOne({self.id_field: record.get(self.id_field)}, {"$set": doc}, upsert=True)
                )
            result = self.collection.bulk_write(operations)
            written = result.upserted_count + result.modified_count
            print(f"[pipeline:inject] upserted {written} docs into '{self.collection_name}'")
        else:
            docs = []
            for record, text, embedding in zip(records, texts, embeddings):
                doc = dict(record)
                doc["embeddedText"] = text
                doc["embedding"] = embedding
                docs.append(doc)
            result = self.collection.insert_many(docs)
            written = len(result.inserted_ids)
            print(f"[pipeline:inject] inserted {written} docs into '{self.collection_name}'")
        return written

    def _ensure_index(self):
        existing = {idx["name"] for idx in self.collection.list_search_indexes()}
        if self.vector_index_name in existing:
            print(f"[pipeline:index] '{self.vector_index_name}' already exists, skipping")
            return
        self.collection.create_search_index({
            "name": self.vector_index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {"type": "vector", "path": "embedding",
                     "numDimensions": self.num_dimensions, "similarity": "cosine"}
                ]
            },
        })
        print(f"[pipeline:index] created '{self.vector_index_name}' ({self.num_dimensions} dims). "
              f"Atlas needs ~1-2 min to finish building it.")

    def run(self, source: Union[str, list]) -> int:
        try:
            records = self._fetch(source)
            if not records:
                print("[pipeline] no records to process, aborting")
                return 0
            texts = self._clean(records)
            embeddings = self._embed(texts)
            written = self._inject(records, texts, embeddings)
            self._ensure_index()
            print(f"[pipeline] DONE -- {written} records live in '{self.collection_name}', "
                  f"searchable via index '{self.vector_index_name}'")
            return written
        except Exception as e:
            print(f"[pipeline] FAILED at some stage: {e}")
            raise
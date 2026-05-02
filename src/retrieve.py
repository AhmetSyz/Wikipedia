from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TOP_K, TOP_K_PER_TYPE
from src.router import route
from src.embed_store import query as vector_query, get_client, get_or_create_collection

def _fetch_by_entity(entity, k):
    client = get_client()
    coll = get_or_create_collection(client)
    res = coll.get(where={"entity": entity}, include=["documents", "metadatas"], limit=k)
    out = []
    if not res["documents"]:
        return out
    for doc, meta in zip(res["documents"], res["metadatas"]):
        out.append({"text": doc, "entity": meta.get("entity",""), "type": meta.get("type",""), "title": meta.get("title",""), "url": meta.get("url",""), "score": 1.0})
    return out

def retrieve(user_query, k=TOP_K):
    route_type, people_hits, place_hits = route(user_query)
    named_entities = people_hits + place_hits
    if route_type == "person":
        chunks = _smart_retrieve(user_query, k, "person", named_entities)
    elif route_type == "place":
        chunks = _smart_retrieve(user_query, k, "place", named_entities)
    else:
        per = max(TOP_K_PER_TYPE, k // 2)
        chunks = _interleave(_smart_retrieve(user_query, per, "person", people_hits), _smart_retrieve(user_query, per, "place", place_hits))
    return {"route": route_type, "matched_people": people_hits, "matched_places": place_hits, "chunks": chunks}

def _smart_retrieve(query, k, type_filter, named_entities):
    if not named_entities:
        return vector_query(query, k=k, type_filter=type_filter)
    per_entity = max(2, k // max(len(named_entities), 1))
    entity_chunks = []
    for entity in named_entities:
        entity_chunks.extend(_fetch_by_entity(entity, per_entity))
    remaining = k - len(entity_chunks)
    entity_names = set(named_entities)
    if remaining > 0:
        semantic = vector_query(query, k=remaining+5, type_filter=type_filter)
        entity_chunks.extend([c for c in semantic if c["entity"] not in entity_names][:remaining])
    return entity_chunks[:k]

def _interleave(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(x)
        out.append(y)
    if len(a) > len(b): out.extend(a[len(b):])
    elif len(b) > len(a): out.extend(b[len(a):])
    return out

"""
Central configuration for the Wikipedia RAG system.

All knobs (entities, models, paths, chunking) live here so the rest of the
codebase stays clean.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Models (all served locally via Ollama)
# --------------------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"            # generation
EMBED_MODEL = "nomic-embed-text"     # embeddings (768-dim)

# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
COLLECTION_NAME = "wikipedia_kb"

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
CHUNK_SIZE = 800       # target characters per chunk
CHUNK_OVERLAP = 120     # characters of overlap between adjacent chunks

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
TOP_K = 5              # chunks per query
TOP_K_PER_TYPE = 3     # for mixed queries: chunks per category

# --------------------------------------------------------------------------
# Required entities (the assignment's minimum set)
# --------------------------------------------------------------------------
PEOPLE = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    # extras to reach 20 and improve coverage
    "Isaac Newton",
    "Charles Darwin",
    "Stephen Hawking",
    "Pablo Picasso",
    "Vincent van Gogh",
    "Mahatma Gandhi",
    "Nelson Mandela",
    "Mustafa Kemal Atatürk",
    "Mozart",
    "Ludwig van Beethoven",
]

PLACES = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Giza pyramid complex",   # canonical Wikipedia title for "Pyramids of Giza"
    "Mount Everest",
    # extras to reach 20 and improve "mixed" coverage (e.g. Turkey)
    "Topkapı Palace",
    "Cappadocia",
    "Stonehenge",
    "Acropolis of Athens",
    "Petra",
    "Christ the Redeemer (statue)",
    "Sydney Opera House",
    "Angkor Wat",
    "Mount Fuji",
    "Niagara Falls",
]
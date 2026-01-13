import json
import os

MAX_RESULTS = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75

def load_movies() -> list[dict]:
    try:
        with open(DATA_PATH, "r") as file:
            data = json.load(file)
        return data["movies"]
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        return []
    
def load_stop_words() -> list[str]:
    try:
        with open(STOP_WORDS_PATH, "r") as file:
            data = file.read()
        return data.splitlines()
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        return []
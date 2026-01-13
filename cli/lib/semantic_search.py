from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Must pass a non-empty text string")
        results = self.model.encode([text])
        return results[0]

    

def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text: str) -> float:
    search = SemanticSearch()
    return search.generate_embedding(text)
import os
import json
import hashlib
import shutil
import numpy as np
import soundfile as sf

class SemanticCacheManager:
    """Manager for caching generated audio segments based on text similarity."""
    
    def __init__(self, cache_dir: str = None, similarity_threshold: float = 0.95):
        if cache_dir is None:
            # Save cache in the storage folder
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.cache_dir = os.path.join(base_dir, "storage", "cache")
        else:
            self.cache_dir = os.path.abspath(cache_dir)
            
        self.similarity_threshold = similarity_threshold
        self.meta_path = os.path.join(self.cache_dir, "metadata.json")
        self.audio_dir = os.path.join(self.cache_dir, "audio")
        
        os.makedirs(self.audio_dir, exist_ok=True)
        
        self.entries = []
        self.index = None
        self.encoder = None
        self.use_neural = False
        
        # Load existing cache metadata
        self._load_metadata()
        
        # Try to initialize neural encoder and FAISS index
        self._init_encoder_and_faiss()

    def _load_metadata(self):
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
                print(f"[CACHE] Loaded {len(self.entries)} entries from cache metadata.")
            except Exception as e:
                print(f"[CACHE WARNING] Could not load metadata: {e}")
                self.entries = []

    def _save_metadata(self):
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CACHE ERROR] Could not save metadata: {e}")

    def _init_encoder_and_faiss(self):
        try:
            # We enforce faiss-cpu==1.7.3 version compatibility
            import faiss
            
            # Try to load SentenceTransformer for neural embeddings
            from sentence_transformers import SentenceTransformer
            print("[CACHE] Initializing SentenceTransformer ('all-MiniLM-L6-v2')...")
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self.use_neural = True
            
            # Rebuild FAISS index from metadata
            dimension = 384 # Dimension of all-MiniLM-L6-v2 embeddings
            self.index = faiss.IndexFlatIP(dimension) # Cosine similarity for normalized vectors
            
            if self.entries:
                vectors = []
                for entry in self.entries:
                    # If entries don't have vectors, we encode them
                    if "vector" in entry:
                        vectors.append(entry["vector"])
                    else:
                        vec = self.encoder.encode(entry["text"], convert_to_numpy=True)
                        # L2 normalization for Cosine Similarity
                        faiss.normalize_L2(vec.reshape(1, -1))
                        entry["vector"] = vec.tolist()
                        vectors.append(entry["vector"])
                
                if vectors:
                    np_vectors = np.array(vectors, dtype=np.float32)
                    self.index.add(np_vectors)
                    print(f"[CACHE] FAISS index rebuilt with {len(self.entries)} vectors.")
            self._save_metadata()
        except ImportError:
            print("[CACHE] sentence-transformers or faiss is not available. Falling back to TF-IDF lexical cache.")
            self.use_neural = False
            self.index = None
            self.encoder = None

    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Fallback character n-gram cosine similarity (Offline, no external weights)."""
        from collections import Counter
        import math
        
        # Helper to extract char 3-grams
        def get_ngrams(text, n=3):
            text = " " + text.lower().strip() + " "
            return [text[i:i+n] for i in range(len(text)-n+1)]
            
        vec1 = Counter(get_ngrams(text1))
        vec2 = Counter(get_ngrams(text2))
        
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
        if not denominator:
            return 0.0
        return float(numerator) / denominator

    def get(self, text: str, temp_output_path: str) -> bool:
        """Checks if a similar sentence is cached. 
        
        If found, copies the cached WAV file to temp_output_path and returns True.
        """
        if not text or not self.entries:
            return False
            
        text_cleaned = text.strip()
        matched_entry = None
        max_sim = 0.0
        
        if self.use_neural and self.index is not None and self.encoder is not None:
            try:
                import faiss
                query_vector = self.encoder.encode(text_cleaned, convert_to_numpy=True)
                faiss.normalize_L2(query_vector.reshape(1, -1))
                
                # Query index for top-1 match
                D, I = self.index.search(np.array([query_vector], dtype=np.float32), 1)
                sim = float(D[0][0])
                idx = int(I[0][0])
                
                if idx >= 0 and idx < len(self.entries) and sim >= self.similarity_threshold:
                    matched_entry = self.entries[idx]
                    max_sim = sim
            except Exception as e:
                print(f"[CACHE WARNING] FAISS search failed: {e}. Falling back to lexical match.")
                matched_entry = None
                
        # Fallback to lexical match
        if matched_entry is None:
            for entry in self.entries:
                sim = self._compute_tfidf_similarity(text_cleaned, entry["text"])
                if sim >= self.similarity_threshold and sim > max_sim:
                    max_sim = sim
                    matched_entry = entry
                    
        if matched_entry and max_sim >= self.similarity_threshold:
            cached_audio = matched_entry["audio_path"]
            if os.path.exists(cached_audio):
                try:
                    # Copy cached file to segment destination
                    os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
                    shutil.copy(cached_audio, temp_output_path)
                    print(f" -> [CACHE HIT] Similarity: {max_sim:.3f} | Text: '{text_cleaned[:40]}...'")
                    return True
                except Exception as e:
                    print(f"[CACHE ERROR] Failed to copy cached file: {e}")
                    
        return False

    def set(self, text: str, temp_segment_path: str):
        """Saves a successfully generated segment file to the cache folder."""
        if not text or not os.path.exists(temp_segment_path):
            return
            
        text_cleaned = text.strip()
        
        # Check if already cached exactly to avoid duplicates
        for entry in self.entries:
            if entry["text"] == text_cleaned:
                return
                
        # Generate a unique hash for the cached WAV file
        hash_val = hashlib.sha256(text_cleaned.encode("utf-8")).hexdigest()
        cached_wav_path = os.path.join(self.audio_dir, f"{hash_val}.wav")
        
        try:
            # Copy file to cache directory before concatenate_wavs deletes it
            shutil.copy(temp_segment_path, cached_wav_path)
            
            entry = {
                "text": text_cleaned,
                "audio_path": cached_wav_path,
                "created_at": os.path.getmtime(cached_wav_path)
            }
            
            # Add embedding if neural encoder is active
            if self.use_neural and self.encoder is not None and self.index is not None:
                import faiss
                vec = self.encoder.encode(text_cleaned, convert_to_numpy=True)
                faiss.normalize_L2(vec.reshape(1, -1))
                entry["vector"] = vec.tolist()
                self.index.add(np.array([vec], dtype=np.float32))
                
            self.entries.append(entry)
            self._save_metadata()
            
        except Exception as e:
            print(f"[CACHE ERROR] Failed to save segment to cache: {e}")

    def clear(self) -> bool:
        """Deletes all cached audio files and clears the metadata index."""
        try:
            # Recreate audio directory
            if os.path.exists(self.audio_dir):
                shutil.rmtree(self.audio_dir)
            os.makedirs(self.audio_dir, exist_ok=True)
            
            # Recreate metadata.json
            self.entries = []
            self._save_metadata()
            
            # Reset FAISS index if initialized
            if self.use_neural and self.index is not None:
                import faiss
                dimension = 384
                self.index = faiss.IndexFlatIP(dimension)
                
            print("[CACHE] Semantic Cache cleared successfully.")
            return True
        except Exception as e:
            print(f"[CACHE ERROR] Failed to clear cache: {e}")
            return False

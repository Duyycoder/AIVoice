import sys
import os

# Set PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.storytelling.context_manager import ContextManager
from app.services.storytelling.image_generator import StorytellingPipeline

print("Initializing Context...")
ctx_mgr = ContextManager("Nguoi_Tren_Van_Nguoi")
ctx = ctx_mgr.load_context()

print("Initializing Pipeline...")
pipe = StorytellingPipeline(ctx)

print("Warming up pipeline (this will download models)...")
try:
    pipe.warmup()
    print("Warmup successful!")
except Exception as e:
    print(f"Exception during warmup: {e}")

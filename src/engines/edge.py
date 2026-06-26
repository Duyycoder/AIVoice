import asyncio
import os
import time
import edge_tts
from src.engines.base import BaseTTSEngine

class EdgeEngine(BaseTTSEngine):
    """Adapter for the Microsoft Edge-TTS online voice engine."""
    
    def __init__(self, voice: str = "vi-VN-NamMinhNeural"):
        self.voice = voice
        
    def generate(self, text: str, output_path: str, **kwargs) -> bool:
        # Check voice overrides
        voice_name = kwargs.get("voice") or self.voice or "vi-VN-NamMinhNeural"
        speed = kwargs.get("speed", 1.0)
        
        # Calculate rate string if speed is not default
        rate = None
        if speed != 1.0:
            pct = int((speed - 1.0) * 100)
            rate = f"+{pct}%" if pct >= 0 else f"{pct}%"
        
        # Ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        async def _save():
            if rate:
                communicate = edge_tts.Communicate(text, voice_name, rate=rate)
            else:
                communicate = edge_tts.Communicate(text, voice_name)
            await communicate.save(output_path)
            
        # Try up to 5 times with exponential backoff to handle rate-limiting or network resets
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Run the asynchronous edge_tts saving synchronously
                asyncio.run(_save())
                
                # Check if the output file was successfully created and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return True
                else:
                    raise ValueError("Output audio file is missing or empty.")
            except Exception as e:
                print(f"EdgeEngine generation attempt {attempt + 1}/{max_retries} failed: {e}")
                # Remove empty or invalid file if it was created
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                if attempt < max_retries - 1:
                    backoff_time = [2.0, 5.0, 10.0, 15.0][attempt]
                    print(f"Rate limit or connection issue. Waiting {backoff_time}s before retry...")
                    time.sleep(backoff_time)
                    
        return False


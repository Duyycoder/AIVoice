import asyncio
import os
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
            
        try:
            # Run the asynchronous edge_tts saving synchronously
            asyncio.run(_save())
            return True
        except Exception as e:
            print(f"EdgeEngine generation failed: {e}")
            return False

from abc import ABC, abstractmethod

class BaseTTSEngine(ABC):
    @abstractmethod
    def generate(self, text: str, output_path: str, **kwargs) -> bool:
        """Generates audio for a single text chunk and saves it.
        
        Args:
            text: The plain text chunk to convert to speech.
            output_path: Path to output WAV segment.
            kwargs: Extra parameters like speed, voice, ref_audio.
            
        Returns:
            bool: Success status (True if successful, False otherwise).
        """
        pass

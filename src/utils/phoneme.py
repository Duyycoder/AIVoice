"""Phonemizer utility for improving Vietnamese pronunciation in XTTSv2.

Uses the viphoneme library to convert raw text into IPA phonemes.
"""

def phonemize_vietnamese(text: str) -> str:
    """Converts Vietnamese text into IPA phonemes.
    
    Falls back to the raw text if viphoneme is not installed or fails.
    """
    try:
        # Monkeypatch vinorm.TTSnorm to return the text unmodified.
        # This bypasses the Linux-compiled binary execution in vinorm on Windows.
        try:
            import vinorm
            vinorm.TTSnorm = lambda t, *a, **kw: t
        except Exception:
            pass

        from viphoneme import vi2IPA
        phonemes = vi2IPA(text)
        if phonemes:
            return phonemes.strip()
        return text
    except ImportError:
        # Graceful fallback when library is missing
        return text
    except Exception as e:
        print(f"Warning: Phonemization failed with error: {e}. Falling back to raw text.")
        return text

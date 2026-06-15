import re

def clean_markdown(text: str) -> str:
    """Strips markdown formatting tags and symbols, returning clean plain text."""
    if not text:
        return ""
        
    # Strip block code (triple backticks)
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Strip inline code (single backticks)
    text = re.sub(r'`([\w\s.,!?;:"\'()\-–—/\\\[\]{}<>@#$%^&*+=|~`]+)`', r'\1', text, flags=re.UNICODE)
    # Strip image markdown ![alt](url) -> empty
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
    # Strip link markdown [text](url) -> text
    text = re.sub(r'\[([\w\s.,!?;:"\'()\-–—/\\\[\]{}<>@#$%^&*+=|~`]+)\]\([^)]+\)', r'\1', text, flags=re.UNICODE)
    # Strip bold and italics formatting (**text**, *text*, __text__, _text_)
    text = re.sub(r'\*\*([\w\s.,!?;:"\'()\-–—/\\\[\]{}<>@#$%^&*+=|~`]+)\*\*', r'\1', text, flags=re.UNICODE)
    text = re.sub(r'\*([\w\s.,!?;:"\'()\-–—/\\\[\]{}<>@#$%^&*+=|~`]+)\*', r'\1', text, flags=re.UNICODE)
    text = re.sub(r'__([\w\s.,!?;:"\'()\-–—/\\\[\]{}<>@#$%^&*+=|~`]+)__', r'\1', text, flags=re.UNICODE)
    text = re.sub(r'_([\w\s.,!?;:"\'()\-–—/\\\[\]{}<>@#$%^&*+=|~`]+)_', r'\1', text, flags=re.UNICODE)
    # Strip headers (e.g. # Header, ## Header)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Strip bullet point markers (e.g. - list, * list)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Strip numbered list markers (e.g. 1. item, 2. item)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    return text.strip()

def chunk_text(text: str, max_words: int = 30) -> list[str]:
    """Splits plain text into manageable sentence/punctuation-bound chunks.
    
    If a sentence is longer than max_words, it is split on commas or spaces
    to prevent memory and length limits of TTS models.
    
    Short sentences are merged together within the same line (paragraph)
    to improve prosody and voice quality, and to speed up processing.
    """
    if not text:
        return []
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    chunks = []
    
    for line in lines:
        # Split by sentence terminals (. ? ! ;), keeping the terminals
        parts = re.split(r'([.?!;]+)', line)
        sentences = []
        
        # Merge parts back into sentence + terminal pairs
        for i in range(0, len(parts) - 1, 2):
            sentences.append(parts[i] + parts[i+1])
        if len(parts) % 2 != 0 and parts[-1]:
            sentences.append(parts[-1])
            
        line_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            words = sentence.split()
            if len(words) <= max_words:
                line_sentences.append(sentence)
            else:
                # Long sentence: split it by commas or spaces
                current_chunk = []
                for word in words:
                    current_chunk.append(word)
                    # Split if chunk is getting large, or if word ends in comma/colon
                    if len(current_chunk) >= max_words or word.endswith(','):
                        line_sentences.append(" ".join(current_chunk))
                        current_chunk = []
                if current_chunk:
                    line_sentences.append(" ".join(current_chunk))
                    
        # Merge consecutive short sentences within the same line
        current_chunk = []
        current_word_count = 0
        for s in line_sentences:
            s_words = s.split()
            s_word_count = len(s_words)
            if not s_words:
                continue
                
            if not current_chunk:
                current_chunk.append(s)
                current_word_count = s_word_count
            elif current_word_count + s_word_count <= max_words:
                current_chunk.append(s)
                current_word_count += s_word_count
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [s]
                current_word_count = s_word_count
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
    return [c.strip() for c in chunks if c.strip()]

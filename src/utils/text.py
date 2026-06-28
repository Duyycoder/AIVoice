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
    # Strip horizontal rules (---, ***, ___)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Strip blockquote indicators (e.g. > quote)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    # Strip checkboxes [ ] and [x]
    text = re.sub(r'\[[ xX]?\]', '', text)
    # Strip headers (e.g. # Header, ## Header)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Strip bullet point markers (e.g. - list, * list)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Strip numbered list markers (e.g. 1. item, 2. item)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    return text.strip()

def chunk_text(text: str, max_words: int = 50, max_chars: int = 240) -> list[str]:
    """Splits plain text into manageable sentence/punctuation-bound chunks.
    
    If a sentence is longer than max_words or max_chars, it is split on commas or spaces
    to prevent memory and length limits of TTS models.
    
    Short sentences are merged together within the same line (paragraph)
    to improve prosody and voice quality, and to speed up processing, 
    but without exceeding max_chars.
    
    Note: For Vietnamese, XTTSv2 has a hard 250-character limit. Enforcing max_chars=240
    avoids CUDA assertion crashes.
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
            # If the sentence is short enough in terms of words AND characters, keep it whole
            if len(words) <= max_words and len(sentence) <= max_chars:
                line_sentences.append(sentence)
            else:
                # Long sentence: split it by commas or spaces
                current_chunk = []
                for word in words:
                    # Test length of chunk with the new word added
                    test_chunk = current_chunk + [word]
                    test_str = " ".join(test_chunk)
                    
                    if (len(test_chunk) > max_words or len(test_str) > max_chars):
                        if current_chunk:
                            line_sentences.append(" ".join(current_chunk))
                            if len(word) > max_chars:
                                word_truncated = word[:max_chars - 3] + "..."
                                line_sentences.append(word_truncated)
                                current_chunk = []
                            else:
                                current_chunk = [word]
                        else:
                            # Single word exceeds max_chars! Truncate it to avoid engine crashes
                            word_truncated = word[:max_chars - 3] + "..."
                            line_sentences.append(word_truncated)
                            current_chunk = []
                    else:
                        current_chunk.append(word)
                        if word.endswith(','):
                            line_sentences.append(" ".join(current_chunk))
                            current_chunk = []
                            
                if current_chunk:
                    line_sentences.append(" ".join(current_chunk))
                    
        # Merge consecutive short sentences within the same line
        current_chunk = []
        current_word_count = 0
        current_char_count = 0
        for s in line_sentences:
            s_words = s.split()
            s_word_count = len(s_words)
            s_char_count = len(s)
            if not s_words:
                continue
                
            if not current_chunk:
                current_chunk.append(s)
                current_word_count = s_word_count
                current_char_count = s_char_count
            else:
                # Character count with space separator
                merged_char_count = current_char_count + 1 + s_char_count
                if current_word_count + s_word_count <= max_words and merged_char_count <= max_chars:
                    current_chunk.append(s)
                    current_word_count += s_word_count
                    current_char_count = merged_char_count
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [s]
                    current_word_count = s_word_count
                    current_char_count = s_char_count
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
    # Filter out chunks that do not contain any pronounceable/alphanumeric characters
    return [c.strip() for c in chunks if c.strip() and any(char.isalnum() for char in c)]

from unicodedata import normalize
from num2words import num2words

_symbols_vi = [
    (re.compile(r'&'), ' và '),
    (re.compile(r'@'), ' a còng '),
    (re.compile(r'%'), ' phần trăm '),
    (re.compile(r'#'), ' số '),
    (re.compile(r'\+'), ' cộng '),
    (re.compile(r'\$'), ' đô la '),
    (re.compile(r'₫'), ' đồng '),
    (re.compile(r'(?<=\d)\s*đ\b'), ' đồng '),
]

_abbreviations_vi = [
    (re.compile(r'\bvcl\b', re.IGNORECASE), 'vờ cờ lờ'),
    (re.compile(r'\bSĐT\b', re.IGNORECASE), 'số điện thoại'),
    (re.compile(r'\bSđt\b', re.IGNORECASE), 'số điện thoại'),
    (re.compile(r'\bsđt\b', re.IGNORECASE), 'số điện thoại'),
    (re.compile(r'\bTP\b\.?\s*', re.IGNORECASE), 'thành phố '),
    (re.compile(r'\bHCM\b', re.IGNORECASE), 'hồ chí minh'),
    (re.compile(r'\bđ/c\b', re.IGNORECASE), 'địa chỉ'),
    (re.compile(r'\bđ\.c\b', re.IGNORECASE), 'địa chỉ'),
    (re.compile(r'\bkm\b', re.IGNORECASE), 'ki lô mét'),
    (re.compile(r'\bm\b', re.IGNORECASE), 'mét'),
    (re.compile(r'\bkg\b', re.IGNORECASE), 'ki lô gam'),
]

def vietnamese_cleaners(text: str) -> str:
    """Specialized text cleaners for Vietnamese.
    
    Performs Unicode NFC normalization, lowercasing, abbreviation expansion,
    symbol replacement, and number spelling (using num2words in Vietnamese)
    while preserving tone/diacritical marks.
    """
    if not text:
        return ""
    text = normalize('NFC', text)
    text = text.lower()
    
    # Expand symbols and abbreviations
    for regex, replacement in _symbols_vi:
        text = re.sub(regex, replacement, text)
    for regex, replacement in _abbreviations_vi:
        text = re.sub(regex, replacement, text)
        
    # Remove thousands separators (e.g. 1.000.000 or 1,000,000)
    text = re.sub(r'\b(\d{1,3})([.,]\d{3})+\b', lambda m: m.group(0).replace('.', '').replace(',', ''), text)
    
    # Expand decimal numbers (e.g. 12.5 or 12,5)
    def repl_decimal(m):
        val = m.group(0).replace(',', '.')
        try:
            return num2words(float(val), lang='vi')
        except:
            return m.group(0)
    text = re.sub(r'\b\d+[.,]\d+\b', repl_decimal, text)
    
    # Expand integer numbers
    def repl_int(m):
        try:
            return num2words(int(m.group(0)), lang='vi')
        except:
            return m.group(0)
    text = re.sub(r'\b\d+\b', repl_int, text)
    
    # Add space around punctuation to prevent G2P/alignment errors
    text = re.sub(r'([a-z0-9_đà-ỹ]+)([.,!?;:…])', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'([.,!?;:…])([a-z0-9_đà-ỹ]+)', r'\1 \2', text, flags=re.IGNORECASE)
    
    # Clean up standard punctuation/symbols
    text = text.replace(';', ',')
    text = text.replace(':', ',')
    text = text.replace('-', ' ')
    text = re.sub(r'[\<\>\(\)\[\]\"]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


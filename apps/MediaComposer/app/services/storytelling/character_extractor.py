import json
import re
from loguru import logger
from typing import List, Dict, Any, Optional

from app.services.llm import get_llm_client

def call_llm(messages: List[dict], temperature: float = 0.4, max_tokens: int = 2500) -> str:
    """Helper to call the LLM and return text."""
    client, model = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return ""

def _parse_json_from_text(text: str) -> Optional[dict]:
    """Extracs a JSON object from a text string that may contain markdown."""
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        json_str = text[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error: {e}")
            return None
    return None

def extract_characters_from_text(text: str, genre: str = "xianxia") -> List[Dict[str, Any]]:
    """
    Phân tích văn bản và bóc tách tối đa 5-10 nhân vật quan trọng nhất.
    """
    system_prompt = f"""Bạn là một chuyên gia phân tích văn học và thiết kế nhân vật, đặc biệt am hiểu thể loại {genre}.
Nhiệm vụ của bạn là đọc đoạn văn bản truyện (được cung cấp bên dưới) và bóc tách tối đa 5 đến 10 nhân vật chính hoặc phụ quan trọng nhất xuất hiện trong đó.

Với mỗi nhân vật, hãy trích xuất:
1. "name": Tên nhân vật (tiếng Việt).
2. "text_description": Vai trò và mô tả ngoại hình sơ bộ dựa CHÍNH XÁC vào nguyên tác (tiếng Việt). Chú ý kỹ đến độ tuổi, vóc dáng, giới tính, phong cách và địa vị. (VD: Không biến nam chính sinh viên thành thư sinh cấp 2 yếu đuối).
3. "search_query": Từ khóa để tìm kiếm ngoại hình nhân vật này trên Google (VD: "ngoại hình nhân vật Tiêu Viêm Đấu Phá Thương Khung").

Chỉ trả về DUY NHẤT một đối tượng JSON theo cấu trúc sau, không kèm bất kỳ lời giải thích nào:
{{
  "characters": [
    {{
      "name": "Tên nhân vật",
      "text_description": "Mô tả chi tiết từ truyện...",
      "search_query": "Từ khóa tìm kiếm"
    }}
  ]
}}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    logger.info("Calling LLM to extract characters from text...")
    response_text = call_llm(messages, temperature=0.3)
    if not response_text:
        return []
        
    data = _parse_json_from_text(response_text)
    if data and "characters" in data:
        return data["characters"]
    
    logger.warning("Failed to extract characters JSON from LLM response.")
    return []

def refine_character_with_web_search(char_name: str, story_name: str, book_desc: str) -> Dict[str, Any]:
    """
    Dùng LLM (với Web Search Grounding) để tìm kiếm và hoàn thiện mô tả nhân vật, kèm link ảnh.
    """
    prompt = f"""Hãy sử dụng tính năng tìm kiếm trên mạng (Google Search) để tra cứu thông tin ngoại hình và hình ảnh của nhân vật "{char_name}" trong truyện "{story_name}".
Bạn có thể tham khảo thêm mô tả ban đầu này: "{book_desc}".

Yêu cầu trả về kết quả dưới dạng một đối tượng JSON có cấu trúc chính xác như sau. KHÔNG viết thêm bất kỳ lời dẫn nào ngoài khối JSON.

1. "description": Mô tả ngoại hình chi tiết bằng tiếng Việt, bám sát nguyên tác (chú ý độ tuổi, vóc dáng, y phục, thần thái). Tránh sai lệch độ tuổi hoặc phong cách.
2. "keywords_en": Các từ khóa tiếng Anh ngăn cách bởi dấu phẩy dùng cho Stable Diffusion (VD: 1boy, handsome, athletic build, short black hair, black robe).
3. "image_urls": Danh sách các link ảnh trực tiếp của nhân vật này từ kết quả tìm kiếm trên mạng (như Wiki Fandom, Baidu, Pinterest...).

Lưu ý: Bạn cũng có thể chèn các link ảnh tìm được dưới dạng Markdown (ví dụ: ![ảnh](url)) ở cuối câu trả lời nếu điều đó giúp bạn hiển thị ảnh dễ dàng hơn (hệ thống sẽ tự động quét lấy link).

Cấu trúc JSON BẮT BUỘC:
{{
  "description": "...",
  "keywords_en": "...",
  "image_urls": ["url1", "url2"]
}}
"""
    
    messages = [{"role": "user", "content": prompt}]
    logger.info(f"Calling LLM to refine character '{char_name}' with web search...")
    response_text = call_llm(messages, temperature=0.4)
    
    result = {
        "description": book_desc,
        "keywords_en": "",
        "image_urls": []
    }
    
    if not response_text:
        return result
        
    data = _parse_json_from_text(response_text)
    if data:
        result["description"] = data.get("description", book_desc)
        result["keywords_en"] = data.get("keywords_en", "")
        result["image_urls"] = data.get("image_urls", [])
    
    # Quét thêm link ảnh dạng markdown hoặc link trực tiếp trong toàn bộ văn bản trả về để đảm bảo không bị sót
    md_urls = re.findall(r'!\[.*?\]\((.*?)\)', response_text)
    plain_urls = re.findall(r'(https?://\S+\.(?:png|jpg|jpeg|webp))', response_text, re.IGNORECASE)
    
    all_urls = result["image_urls"] + md_urls + plain_urls
    # Lọc trùng lặp và link rỗng
    unique_urls = []
    for u in all_urls:
        u = u.strip('\'"()')
        if u and u not in unique_urls:
            unique_urls.append(u)
            
    result["image_urls"] = unique_urls
    return result

def process_chapters_and_extract_characters(
    chapter_texts: List[str], 
    story_name: str, 
    genre: str, 
    enable_web_search: bool = True
) -> List[Dict[str, Any]]:
    """
    Luồng điều phối chính:
    1. Gộp văn bản (cắt bớt nếu quá dài).
    2. Gọi bóc tách nhân vật.
    3. Trích xuất chi tiết qua Web Search (nếu bật).
    """
    full_text = "\n\n".join(chapter_texts)
    # Giới hạn số lượng ký tự để tránh vượt quá context window (vd: ~80k ký tự)
    if len(full_text) > 80000:
        full_text = full_text[:80000] + "\n...[Nội dung đã được cắt bớt để đảm bảo giới hạn xử lý]..."
        
    logger.info("Bắt đầu bóc tách nhân vật từ văn bản...")
    initial_chars = extract_characters_from_text(full_text, genre)
    
    if not initial_chars:
        return []
        
    final_chars = []
    for char in initial_chars:
        name = char.get("name", "")
        desc = char.get("text_description", "")
        if not name:
            continue
            
        if enable_web_search:
            refined = refine_character_with_web_search(name, story_name, desc)
            final_chars.append({
                "name": name,
                "description": refined.get("description", desc),
                "keywords_en": refined.get("keywords_en", ""),
                "image_urls": refined.get("image_urls", [])
            })
        else:
            final_chars.append({
                "name": name,
                "description": desc,
                "keywords_en": "1boy/1girl, detailed face", # Placeholder, có thể sinh bằng LLM nhanh nếu cần
                "image_urls": []
            })
            
    return final_chars

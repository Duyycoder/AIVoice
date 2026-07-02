import os
import sys

# Add the project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.storytelling.character_extractor import extract_characters_from_text, refine_character_with_web_search

sample_text = """
Hàn Lập ngước mắt nhìn lên bầu trời xám xịt. Hắn là một thanh niên hai mươi tuổi, thân hình gầy gò nhưng ánh mắt cực kỳ kiên định. 
Khoác trên mình bộ thanh sam cũ kỹ, hắn đứng cạnh Nam Cung Uyển - một nữ tử tuyệt sắc với dung nhan lạnh lùng, mặc bạch y bồng bềnh như tiên nữ giáng trần.
Hai người đang chuẩn bị bước vào Huyết Sắc Cấm Địa.
"""

def run_test():
    print("Testing extract_characters_from_text...")
    chars = extract_characters_from_text(sample_text, genre="xianxia")
    print(f"Extracted: {chars}")
    
    if chars:
        char_name = chars[0].get("name")
        char_desc = chars[0].get("text_description")
        print(f"\nTesting refine_character_with_web_search for {char_name}...")
        refined = refine_character_with_web_search(char_name, "Phàm Nhân Tu Tiên", char_desc)
        print(f"Refined: {refined}")

if __name__ == "__main__":
    run_test()

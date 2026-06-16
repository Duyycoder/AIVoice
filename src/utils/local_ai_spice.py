import os
import sys
import gc

def add_spice_to_text_local(text: str, style: str = 'teu_tao', model_path: str = None) -> str:
    """
    Rewrites the input text using a local GGUF LLM to inject emotion and humor (spicing)
    based on the requested style, then cleans up GPU memory to fit within a strict 6GB VRAM limit.
    """
    if not model_path:
        raise ValueError("model_path must be specified to use the local LLM.")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local LLM model not found at: {model_path}")

    # Styles lookup table for Vietnamese rewriting
    styles_prompts = {
        'teu_tao': (
            "Bạn là một biên kịch vui tính, có khiếu hài hước tự nhiên. "
            "Hãy viết lại đoạn văn bản tiếng Việt sau đây theo phong cách tếu táo, hài hước, dí dỏm. "
            "Giữ nguyên ý nghĩa cốt lõi của đoạn văn gốc. "
            "Hãy lồng ghép khéo léo các từ ngữ nói chuyện giao tiếp tiếng Việt nói hàng ngày tự nhiên "
            "(ví dụ: 'nha', 'nhé', 'haha', 'anh em', 'nè', 'ghê', 'vậy ta', 'ha', 'bà con'). "
            "Đặc biệt lưu ý: Chỉ trả về đoạn văn bản đã viết lại, KHÔNG thêm bất kỳ lời dẫn nhập, giải thích, "
            "hoặc dấu nháy kép bọc ngoài nào."
        ),
        'hai_huoc': (
            "Bạn là một diễn viên hài độc thoại. Hãy viết lại đoạn văn bản tiếng Việt sau đây "
            "để trở nên cực kỳ hài hước, hóm hỉnh và thu hút người nghe. "
            "Giữ nguyên nội dung cốt lõi nhưng dùng lối nói dí dỏm, sử dụng các từ ngữ thân mật nói chuyện tự nhiên (e.g., 'haha', 'chèn ơi', 'nha', 'nhé'). "
            "Đặc biệt lưu ý: Chỉ trả về nội dung đã viết lại, không giải thích hay dẫn dắt thêm."
        ),
        'than_thien': (
            "Hãy đóng vai một người bạn thân thiết. Viết lại đoạn văn bản tiếng Việt sau đây "
            "theo phong cách thân thiện, ấm áp và gần gũi nhất có thể. "
            "Sử dụng các từ ngữ nói chuyện nhẹ nhàng, tự nhiên như trò chuyện trực tiếp (e.g., 'nha', 'nhé', 'nè', 'dạ'). "
            "Giữ nguyên nội dung cốt lõi của văn bản. "
            "Đặc biệt lưu ý: Chỉ trả về nội dung đã viết lại, không giải thích hay dẫn dắt thêm."
        ),
    }

    # Fetch corresponding prompt instruction or construct a default one
    system_instruction = styles_prompts.get(
        style,
        f"Hãy viết lại đoạn văn bản sau đây bằng tiếng Việt theo phong cách {style}. "
        "Giữ nguyên ý nghĩa cốt lõi nhưng làm cho câu từ tự nhiên, sinh động hơn bằng cách thêm "
        "các từ ngữ giao tiếp nói hàng ngày (ví dụ: 'nha', 'nhé', 'nè'). "
        "Đặc biệt lưu ý: Chỉ trả về kết quả đã được viết lại, không thêm lời dẫn hay giải thích nào."
    )

    llm = None
    try:
        print(f"Loading local GGUF LLM from {model_path}...")
        # Import dynamically so that llama-cpp-python is only required if spicing is actually requested
        from llama_cpp import Llama
        
        # Instantiate LLM offloading all layers to GPU to maximize hardware acceleration
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,  # Offload all layers to GPU
            verbose=False     # Disable verbose llama.cpp logs to keep console clean
        )

        print("Generating spiced text...")
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        rewritten_text = response["choices"][0]["message"]["content"].strip()

        # Post-process to strip markdown code fences if the model generated them
        if rewritten_text.startswith("```"):
            lines = rewritten_text.splitlines()
            filtered_lines = [line for line in lines if not line.strip().startswith("```")]
            rewritten_text = "\n".join(filtered_lines).strip()
            
        return rewritten_text

    finally:
        # CRITICAL VRAM MANAGEMENT:
        # Explicitly delete the Llama instance to release handle
        if llm is not None:
            print("Releasing local LLM instance...")
            del llm
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA VRAM if PyTorch is available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA VRAM successfully cleared.")
        except ImportError:
            pass

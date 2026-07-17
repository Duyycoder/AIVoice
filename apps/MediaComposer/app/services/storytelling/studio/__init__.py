# -*- coding: utf-8 -*-
"""Studio Compositing — render ảnh cảnh theo lớp (nền + nhân vật) rồi ghép.

Nhánh: feat/studio-compositing. Bật qua config `render_mode="studio"`.
Các module thuần (matting, compositor, layout_planner, background_renderer) KHÔNG
import torch ở top-level → test được không cần GPU. Renderer thật (character/background)
import Stable Diffusion lazy bên trong hàm.
"""

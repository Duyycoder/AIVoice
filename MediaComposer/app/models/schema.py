from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel


class MaterialInfo(BaseModel):
    provider: str = ""
    url: str = ""
    duration: float = 0.0


class VideoAspect(str, Enum):
    portrait = "9:16"
    landscape = "16:9"
    square = "1:1"

    def to_resolution(self):
        if self == VideoAspect.landscape or self.value == "16:9":
            return 1920, 1080
        elif self == VideoAspect.portrait or self.value == "9:16":
            return 1080, 1920
        elif self == VideoAspect.square or self.value == "1:1":
            return 1080, 1080
        else:
            raise ValueError(f"no resolution for {self}")


class VideoConcatMode(str, Enum):
    random = "random"
    sequential = "sequential"


class VideoTransitionMode(str, Enum):
    none = "none"
    fade_in = "fade_in"
    fade_out = "fade_out"
    slide_in = "slide_in"
    slide_out = "slide_out"
    shuffle = "shuffle"


class VideoParams(BaseModel):
    video_aspect: str = "9:16"
    video_concat_mode: str = "random"
    voice_volume: float = 1.0
    bgm_type: str = ""
    bgm_file: str = ""
    bgm_volume: float = 0.0
    subtitle_enabled: bool = True
    font_name: str = "STHeitiMedium.ttc"
    font_size: int = 60
    text_fore_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: float = 1.5
    text_background_color: Optional[Union[str, bool]] = False
    subtitle_bg_alpha: int = 140
    rounded_subtitle_background: bool = False
    subtitle_position: str = "bottom"
    custom_position: float = 70.0
    n_threads: Optional[int] = None

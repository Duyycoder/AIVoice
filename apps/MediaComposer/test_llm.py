import asyncio
from app.services.storytelling.models import Scene, StoryContext, Character
from app.services.storytelling.llm_prompter import generate_prompts_batch

scenes = [
    Scene(scene_id=0, text_vi='Nữ tử áo trắng kia Lạc Lan Tuyết cũng đem ánh mắt nhìn về phía Dịch Phong.', word_count=20, start_time=0.0, end_time=2.0, duration_sec=2.0, image_prompt='', characters_in_scene=[], primary_character='', fallback_level=0, accepted_seed=0, frame_path=''),
    Scene(scene_id=1, text_vi='Tóm lại, để người nhìn lên cực kỳ dễ chịu. Đáng tiếc, là cái phàm nhân. Lạc Lan Tuyết theo đó thu hồi ánh mắt, mặt lộ một chút khinh thường.', word_count=20, start_time=0.0, end_time=2.0, duration_sec=2.0, image_prompt='', characters_in_scene=[], primary_character='', fallback_level=0, accepted_seed=0, frame_path=''),
    Scene(scene_id=2, text_vi='Nhưng để hắn không có nghĩ tới là, hắn trời sinh kinh mạch bế tắc, trọn vẹn cũng không phải là tu luyện liệu. Dịch Phong lắc đầu bất đắc dĩ, quơ quơ bụi bặm trên người, cầm lấy gốm sứ ấm nước, hướng ghế nằm nằm xuống.', word_count=30, start_time=0.0, end_time=2.0, duration_sec=2.0, image_prompt='', characters_in_scene=[], primary_character='', fallback_level=0, accepted_seed=0, frame_path='')
]

context = StoryContext(
    story_name='Nguoi Tren Van Nguoi',
    story_slug='Nguoi_Tren_Van_Nguoi',
    genre='xianxia',
    characters=[
        Character(name='Dịch Phong', slug='d_ch_phong', description='Nam chính...', keywords_en='1boy, silver hair', has_embedding=True),
        Character(name='Lạc Lan Tuyết', slug='l_c_lan_tuy_t', description='Nữ chính...', keywords_en='1girl, white hanfu', has_embedding=True)
    ]
)

generate_prompts_batch(scenes, context, batch_size=3)
for s in scenes:
    print(f'Scene {s.scene_id}: {s.image_prompt}')

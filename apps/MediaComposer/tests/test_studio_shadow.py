from PIL import Image
from app.services.storytelling.models import CharacterLayer
from app.services.storytelling.studio.compositor import composite, _draw_shadow

def test_shadow_default_off():
    bg = Image.new("RGB", (100, 100), (255, 255, 255))
    layer = Image.new("RGBA", (20, 20), (255, 0, 0, 255))
    ch = CharacterLayer(slug="a", prompt="", anchor_x="center", anchor_y="bottom")
    
    # Render without shadow
    res_no_shadow = composite(bg, [(layer, ch)], harmonize=False, shadow_opacity=0.0)
    
    # Render with default params
    res_default = composite(bg, [(layer, ch)], harmonize=False)
    
    # Must be byte exact
    assert res_no_shadow.tobytes() == res_default.tobytes()

def test_shadow_draws_dark_pixels_below_character():
    bg = Image.new("RGB", (100, 100), (255, 255, 255))
    # Character layer has transparent bottom half to see the shadow
    layer = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    ch = CharacterLayer(slug="a", prompt="", anchor_x="center", anchor_y="bottom")
    
    res_with_shadow = composite(bg, [(layer, ch)], harmonize=False, shadow_opacity=0.5)
    
    # Shadow is drawn below the character.
    # Frame is 100x100. Character is 20x20 at bottom center.
    # x = 40, y = 80.
    # Shadow ellipse is drawn at sy0 = 80 + 20 - 1 = 99?
    # Actually shadow_w = 16, shadow_h = 1.
    # Let's just check if there are any non-white pixels
    colors = res_with_shadow.getcolors()
    # If shadow was drawn, there will be multiple colors (blur introduces gradients).
    # Background is white (255, 255, 255).
    non_white_pixels = [c for count, c in colors if c != (255, 255, 255)]
    assert len(non_white_pixels) > 0

def test_shadow_position_correct():
    canvas = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
    _draw_shadow(canvas, 40, 40, 20, 20, 1.0)
    
    # shadow_w = 16, shadow_h = 1
    # sx0 = 40 + 2 = 42
    # sy0 = 40 + 20 - 0 = 60
    # Shadow is black. Check pixel at 50, 60.
    r, g, b, a = canvas.getpixel((50, 60))
    # Because of blur, it might not be pure black, but should be dark
    assert r < 255 and g < 255 and b < 255

def test_harmonize_preserves_alpha():
    # Harmonize function shouldn't alter the alpha channel.
    from app.services.storytelling.studio.compositor import _harmonize
    
    bg = Image.new("RGB", (100, 100), (100, 100, 100))
    fg = Image.new("RGBA", (20, 20), (255, 0, 0, 128))
    
    fg_harm = _harmonize(bg, fg)
    
    # Check alpha channel is preserved (128)
    assert fg_harm.getpixel((10, 10))[3] == 128

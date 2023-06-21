from pathlib import Path
import numpy as np

from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype(
    str(Path(__file__).parent.parent / "FreeMonoBoldOblique.otf"), 15)
# font = ImageFont.truetype(('./Roboto-Medium.ttf'), 32)


def draw_points_on_image(image, points, curr_point=None, highlight_all=True, radius_scale=0.01):
    overlay_rgba = Image.new("RGBA", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    for point_key, point in points.items():
        if ((curr_point is not None and curr_point == point_key)
                or highlight_all):
            p_color = (255, 0, 0)
            t_color = (0, 0, 255)

        else:
            p_color = (255, 0, 0, 35)
            t_color = (0, 0, 255, 35)

        rad_draw = int(image.size[0] * radius_scale)

        p_start = point.get("start_temp", point["start"])
        p_target = point["target"]

        if p_start is not None and p_target is not None:
            p_draw = int(p_start[0]), int(p_start[1])
            t_draw = int(p_target[0]), int(p_target[1])

            overlay_draw.line(
                (p_draw[0], p_draw[1], t_draw[0], t_draw[1]),
                fill=(255, 255, 0),
                width=2,
            )

        if p_start is not None:
            p_draw = int(p_start[0]), int(p_start[1])
            overlay_draw.ellipse(
                (
                    p_draw[0] - rad_draw,
                    p_draw[1] - rad_draw,
                    p_draw[0] + rad_draw,
                    p_draw[1] + rad_draw,
                ),
                fill=p_color,
            )

            if curr_point is not None and curr_point == point_key:
                # overlay_draw.text(p_draw, "p", font=font, align="center", fill=(0, 0, 0))
                overlay_draw.text(p_draw, "p", align="center", fill=(0, 0, 0))

        if p_target is not None:
            t_draw = int(p_target[0]), int(p_target[1])
            overlay_draw.ellipse(
                (
                    t_draw[0] - rad_draw,
                    t_draw[1] - rad_draw,
                    t_draw[0] + rad_draw,
                    t_draw[1] + rad_draw,
                ),
                fill=t_color,
            )

            if curr_point is not None and curr_point == point_key:
                # overlay_draw.text(t_draw, "t", font=font, align="center", fill=(0, 0, 0))
                overlay_draw.text(t_draw, "t", align="center", fill=(0, 0, 0))

    return Image.alpha_composite(image.convert("RGBA"), overlay_rgba).convert("RGB")


def draw_mask_on_image(image, mask):
    im_mask = np.uint8(mask * 255)
    im_mask_rgba = np.concatenate(
        (
            np.tile(im_mask[..., None], [1, 1, 3]),
            45 * np.ones((im_mask.shape[0],
                         im_mask.shape[1], 1), dtype=np.uint8),
        ),
        axis=-1,
    )
    im_mask_rgba = Image.fromarray(im_mask_rgba).convert("RGBA")

    return Image.alpha_composite(image.convert("RGBA"), im_mask_rgba).convert("RGB")


def on_change_single_global_state(keys, value, global_state, map_transform=None):
    if map_transform is not None:
        value = map_transform(value)

    curr_state = global_state
    if isinstance(keys, str):
        last_key = keys

    else:
        for k in keys[:-1]:
            curr_state = curr_state[k]

        last_key = keys[-1]

    curr_state[last_key] = value
    return global_state


def get_latest_points_pair(points_dict):
    if not points_dict:
        return None
    point_idx = list(points_dict.keys())
    latest_point_idx = max(point_idx)
    return latest_point_idx


def make_watermark(image: Image.Image):

    font_size = int(image.size[0] / 512 * 15)
    buffer_size = int(image.size[0] / 512 * 5)
    font = ImageFont.truetype(
        str(Path(__file__).parent.parent / "FreeMonoBoldOblique.otf"),
        font_size)

    text = Image.new("RGBA", image.size, (255, 255, 255, 0))
    canvas = ImageDraw.Draw(text)

    watermark_size = canvas.textbbox((0, 0), 'AI Generated', font)

    watermark_l = watermark_size[2] - watermark_size[0]
    watermark_h = watermark_size[3] - watermark_size[1]

    mean_img_color = np.array(image).mean()
    canvas.rectangle((
            image.size[0] - watermark_l - buffer_size, 
            image.size[1] - watermark_h - buffer_size,
            image.size[0], 
            image.size[1],
        ),
                     fill=(255, 255, 255, 255))
    canvas.text((image.size[0] - watermark_l - buffer_size, 
                 image.size[1] - watermark_h - buffer_size),
                "AI Generated", font=font,
                fill=(0, 0, 0, 255))
    out = Image.alpha_composite(image.convert('RGBA'), text)
    return out

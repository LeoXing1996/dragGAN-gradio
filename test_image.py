from PIL import Image, ImageDraw
import numpy as np
import gradio as gr
from gradio_utils import get_valid_mask


class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload",
                         tool="sketch",
                         interactive=False,
                         **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == "sketch" and self.source in ["upload", "webcam"
                                                     ] and type(x) != dict:
            decode_image = gr.processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.zeros((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        return super().preprocess(x)


def get_latest_points_pair(points_dict):
    if not points_dict:
        return None
    point_idx = list(points_dict.keys())
    latest_point_idx = max(point_idx)
    return latest_point_idx


def update_image_draw(image, points, mask, show_mask):

    image_draw = draw_points_on_image(image, points)
    if show_mask:
        image_draw = draw_mask_on_image(image_draw, mask)
    return image_draw


def draw_points_on_image(image,
                         points,
                         curr_point=None,
                         highlight_all=True,
                         radius_scale=0.01):
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

    return Image.alpha_composite(image.convert("RGBA"),
                                 overlay_rgba).convert("RGB")


def draw_mask_on_image(image, mask):
    if mask is None:
        return image
    im_mask = np.uint8(mask * 255)
    im_mask_rgba = np.concatenate(
        (
            np.tile(im_mask[..., None], [1, 1, 3]),
            45 * np.ones(
                (im_mask.shape[0], im_mask.shape[1], 1), dtype=np.uint8),
        ),
        axis=-1,
    )
    im_mask_rgba = Image.fromarray(im_mask_rgba).convert("RGBA")

    return Image.alpha_composite(image.convert("RGBA"),
                                 im_mask_rgba).convert("RGB")


def on_click_image(global_state, evt: gr.SelectData):
    """This function only support click for point selection
    """
    xy = evt.index
    if global_state['editing_state'] != 'add_points':
        print(f'In {global_state["editing_state"]} state. '
              'Do not add points.')
        return global_state, global_state["images"]["show"]

    points = global_state["points"]

    point_idx = get_latest_points_pair(points)
    if point_idx is None:
        points[0] = {'start': xy, 'target': None}
        print(f'Click Image - Start - {xy}')
    elif points[point_idx].get('target', None) is None:
        points[point_idx]['target'] = xy
        print(f'Click Image - Target - {xy}')
    else:
        points[point_idx + 1] = {'start': xy, 'target': None}
        print(f'Click Image - Start - {xy}')

    image_raw = global_state["images"]["image_raw"]
    image_draw = draw_points_on_image(
        image_raw,
        global_state["points"],
        None,  # NOTE: we hight light all points
        # global_state["curr_point"],
    )
    if global_state['show_mask']:
        image_draw = draw_mask_on_image(image_draw,
                                        global_state.get('mask', None))
        # global_state['image']['show'] = image_draw
    global_state['images']['show'] = image_draw

    # return global_state, image_draw
    return global_state, image_draw


with gr.Blocks() as b:
    global_state = gr.State({
        'images': {
            # the original image, change when seed / model is changes
            'image_ori': Image.open('./raw.png'),
            # image without mask and points, change during optimization
            'image_raw': Image.open('./raw.png'),
            'show': Image.open('./raw.png'),
        },
        'draws': {},
        'editing_state': 'add_points',
        'points': dict(),
        'mask': None,
        # 'show_points': True,
        'show_mask': True,
    })

    image = ImageMask(value=global_state.value['images']['show'],
                      brush_radius=20).style(height=512, width=512)
    add_point = gr.Button('Add Points')
    add_mask = gr.Button('Add Mask')
    show_mask = gr.Checkbox(label='Show Mask',
                            show_label=False,
                            value=global_state.value['show_mask'])

    def on_click_add_point(global_state, image: dict):
        """Function switch from add mask mode to add points mode.
        1. Save mask buffer
        2. Change global_state['editing_state'] to 'add_points'
        3. Set current image with mask
        """
        global_state['editing_state'] = 'add_points'
        global_state['mask'] = get_valid_mask(image['mask'])
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'],
                                       global_state['show_mask'])

        return (global_state,
                gr.Image.update(value=image_draw, interactive=False))

    def on_click_enable_draw(global_state, image: dict):
        """Function switch from add point mode to add mask mode.
        1. Change editing state to add_mask
        2. Set curr image with points and mask
        """
        global_state['editing_state'] = 'add_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True)
        global_state['images']['show'] = image_draw
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    def on_change_show_mask(global_state, image, show_mask):
        """Function to show mask in **add_point mode**.
        1. Change global state
        2. update image of is add poitn mode.
        """
        global_state['show_mask'] = show_mask
        if global_state['editing_state'] != 'add_points':
            return global_state, image
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], show_mask)
        return global_state, image_draw

    add_point.click(on_click_add_point,
                    inputs=[global_state, image],
                    outputs=[global_state, image])
    add_mask.click(on_click_enable_draw,
                   inputs=[global_state, image],
                   outputs=[global_state, image])
    show_mask.change(on_change_show_mask,
                     inputs=[global_state, image, show_mask],
                     outputs=[global_state, image])
    image.select(
        on_click_image,
        inputs=[global_state],
        outputs=[global_state, image],
    )

    # disable_draw.click(on_click_disable_draw, inputs=[image])
    # enable_draw.click(on_click_enable_draw, inputs=[image])

b.launch()

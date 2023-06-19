from ipdb import iex
from functools import partial
from tempfile import NamedTemporaryFile

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

import dnnlib
from gradio_utils import (draw_mask_on_image, draw_points_on_image,
                          on_change_single_global_state,
                          get_latest_points_pair)
from viz.renderer_pickable import Renderer

device = 'cuda'
init_pkl = './checkpoints/stylegan2-cat-config-f.pkl'
init_key = ''


def create_images(image_raw, global_state, update_original=False):
    # import ipdb
    # ipdb.set_trace()
    if isinstance(image_raw, torch.Tensor):
        image_raw = image_raw.cpu().numpy()
        image_raw = Image.fromarray(image_raw)

    if update_original:
        global_state["images"]["image_orig"] = image_raw.copy()
        print('Update image_orig to image_raw')

    global_state["images"]["image_raw"] = image_raw
    show_state = global_state['show_state']

    # global_state['draws']['image_with_points'] = image_raw.copy()
    image_to_draw = image_raw.copy()

    if show_state in ['both', 'point']:
        image_to_draw = draw_points_on_image(
            image_to_draw, global_state["points"],
            global_state["curr_point"])

    if show_state in ['both', 'mask']:
        # NOTE: mask mayby None at start
        if 'image_mask' not in global_state['images']:
            global_state["images"]["image_mask"] = np.ones(
                (image_raw.size[1], image_raw.size[0]), dtype=np.uint8)
        image_to_draw = draw_mask_on_image(
            image_to_draw, global_state['images']['image_mask'])

    global_state['draws']['image_with_points'] = image_to_draw

    # global_state["images"]["image_mask"] = np.ones(
    #     (image_raw.size[1], image_raw.size[0]), dtype=np.uint8)
    # global_state["draws"]["image_with_mask"] = draw_mask_on_image(
    #     global_state["images"]["image_raw"],
    #     global_state["images"]["image_mask"])


with gr.Blocks() as app:

    # renderer = Renderer()
    global_state = gr.State({
        "images": {
            # image_orig
            # image_raw
            # image_mask
        },
        "draws": {
            # image_with_points
            # image_with_mask
        },
        "temporal_params": {
            # stop
        },
        "generator_params": dnnlib.EasyDict(),
        "params": {
            "seed": 4211,  # cute cat ^_^
            "motion_lambda": 20,
            "r1_in_pixels": 3,
            "r2_in_pixels": 12,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": 1e-3,
        },
        "device": device,
        "draw_interval": 5,
        "radius_mask": 51,
        "renderer": Renderer(),
        "points": {},
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',  # only edit points as default
        'show_state': 'both',  # show mask and points as default
    })

    # init image
    global_state.value['renderer'].init_network(
        global_state.value['generator_params'],  # res
        init_pkl,  # pkl
        global_state.value['params']['seed'],  # w0_seed,
        global_state.value['params']['latent_space'] == 'w+',  # w_plus
        'const',
        global_state.value['params']['trunc_psi'],  # trunc_psi,
        None,  # trunc_cutoff,
        None,  # input_transform
        global_state.value['params']['lr']  # lr,
    )
    global_state.value['renderer']._render_drag_impl(
        global_state.value['generator_params'], is_drag=False)
    create_images(global_state.value['generator_params'].image,
                  global_state.value, update_original=True)

    with gr.Row():
        # Left column
        with gr.Column(scale=0.7):

            with gr.Accordion("Network & latent"):

                with gr.Row():
                    with gr.Tab("Pretrained models"):
                        form_pretrained_dropdown = gr.Dropdown(
                            choices=list(global_state.value['renderer'].
                                         PRETRAINED_MODELS.keys()),
                            label="Pretrained model",
                            value="afhqcat",
                        )

                    with gr.Tab("Local file"):
                        form_model_pickle_file = gr.File(label="Pickle file")

                    with gr.Tab("URL"):
                        with gr.Row():
                            form_model_url = gr.Textbox(
                                placeholder="Url of the pickle file",
                                label="URL",
                            )
                            form_model_url_btn = gr.Button("Submit")

                with gr.Row().style(equal_height=True):
                    with gr.Tab("Image seed"):
                        with gr.Row():
                            form_seed_number = gr.Number(
                                value=42,
                                interactive=True,
                                label="Seed",
                            )
                            form_update_image_seed_btn = gr.Button(
                                "Update image")

                    with gr.Tab("Image projection"):
                        with gr.Row():
                            form_project_file = gr.File(
                                label="Image project file")
                            form_project_iterations_number = gr.Number(
                                value=1_000,
                                label="Image projection num steps",
                            )
                            form_update_image_project_btn = gr.Button(
                                "Run projection")

                    form_reset_image = gr.Button("Reset image")

        with gr.Accordion('Tools'):
            enable_add_points = gr.Button('Draw Points')
            # TODO: support mask
            # form_enable_add_mask = gr.Button('Draw Mask')
            undo_points = gr.Button('Undo')
            enable_add_mask = gr.Button('Draw Mask')

            # TODO: add a list to show all control points

            form_reset_mask_btn = gr.Button("Reset mask").style(
                full_width=True)
            form_radius_mask_number = gr.Number(
                value=global_state.value["radius_mask"],
                interactive=True,
                label="Radius (pixels)",
            ).style(full_width=False)

        with gr.Row():
            # with gr.Tab("Run"):
            with gr.Row():
                with gr.Column():
                    form_start_btn = gr.Button("Start").style(
                        full_width=True)
                    form_stop_btn = gr.Button("Stop").style(
                        full_width=True)
                form_steps_number = gr.Number(
                    value=0, label="Steps",
                    interactive=False).style(full_width=False)
                form_draw_interval_number = gr.Number(
                    value=global_state.value["draw_interval"],
                    label="Draw Interval (steps)",
                    interactive=True,
                ).style(full_width=False)
                form_download_result_file = gr.File(
                    label="Download result",
                    visible=False).style(full_width=True)

            # with gr.Tab("Hyperparameters"):
        with gr.Row():
            form_lambda_number = gr.Number(
                value=global_state.value["params"]["motion_lambda"],
                interactive=True,
                label="Lambda",
            ).style(full_width=True)
            with gr.Column():
                form_lr_number = gr.Number(
                    value=global_state.value["params"]["lr"],
                    interactive=True,
                    label="LR",
                ).style(full_width=True)
                update_lr_number = gr.Button('Update')
            form_magnitude_direction_in_pixels_number = gr.Number(
                value=global_state.value["params"]
                ["magnitude_direction_in_pixels"],
                interactive=True,
                label=("Magnitude direction of d vector"
                        " (pixels)"),
            ).style(full_width=True)

        # with gr.Row():
            form_r1_in_pixels_number = gr.Number(
                value=global_state.value["params"]["r1_in_pixels"],
                interactive=True,
                label="R1 (pixels)",
            ).style(full_width=False)
            form_r2_in_pixels_number = gr.Number(
                value=global_state.value["params"]["r2_in_pixels"],
                interactive=True,
                label="R2 (pixels)",
            ).style(full_width=False)

            form_latent_space = gr.Radio(
                ['w', 'w+'],
                value=global_state.value['params']['latent_space'],
                interactive=True,
                label='Latent space to optimize',
            )

        with gr.Column():
            # form_image_draw = gr.Image(
            #     global_state.value["draws"]["image_with_points"],
            #     elem_classes="image_nonselectable")
            form_image_draw = gr.Image(
                global_state.value["draws"]["image_with_points"],
                tool='sketch',
                interactive=False)
            # form_image_mask_draw = gr.Image(
            #     global_state.value["draws"]["image_with_mask"],
            #     visible=False,
            #     elem_classes="image_nonselectable",
            # )
            gr.Markdown(
                "Credits: Adrià Ciurana Lanau | info@dreamlearning.ai | OpenMMLab | ?"
            )

        # Network & latents tab listeners
        def on_change_pretrained_dropdown(pretrained_value, global_state,
                                          seed):
            renderer: Renderer = global_state["renderer"]

            renderer.init_network(
                global_state['generator_params'],  # res
                pretrained_value,  # pkl
                global_state['params']['seed'],  # w0_seed,
                global_state['params']['latent_space'] == 'w+',  # w_plus
                'const',
                global_state['params']['trunc_psi'],  # trunc_psi,
                global_state['params']['trunc_cutoff'],  # trunc_cutoff,
                None,
                global_state['params']['lr']  # lr,
            )

            renderer._render_drag_impl(global_state['generator_params'],
                                       is_drag=False)
            image_raw = global_state['generator_params'].image
            create_images(image_raw, global_state, update_original=True)

            # return global_state, image_raw, global_state["draws"][
            #     "image_with_mask"]
            return global_state, image_raw

        form_pretrained_dropdown.change(
            on_change_pretrained_dropdown,
            inputs=[form_pretrained_dropdown, global_state, form_seed_number],
            # outputs=[global_state, form_image_draw, form_image_mask_draw],
            outputs=[global_state, form_image_draw],
        )

        def on_click_reset_image(global_state):
            global_state["images"]["image_raw"] = global_state["images"][
                "image_orig"].copy()
            global_state["draws"]["image_with_points"] = global_state[
                "images"]["image_orig"].copy()
            global_state['points'] = dict()
            print('Clear points')

            global_state['images']['image_with_mask'] = np.ones(
                (
                    global_state["images"]["image_raw"].size[1],
                    global_state["images"]["image_raw"].size[0],
                ),
                dtype=np.uint8,
            )
            print('Clear mask')

            # return global_state, global_state["images"][
            #     "image_raw"], global_state["draws"]["image_with_mask"]
            return global_state, global_state["images"]["image_raw"]

        form_reset_image.click(
            on_click_reset_image,
            inputs=[global_state],
            # outputs=[global_state, form_image_draw, form_image_mask_draw],
            outputs=[global_state, form_image_draw],
        )

        # Update parameters
        def on_change_update_image_seed(seed, global_state):

            renderer = global_state["renderer"]
            global_state["params"]["seed"] = int(seed)
            renderer.init_network(
                global_state['generator_params'],  # res
                renderer.pkl,  # pkl
                # pretrained_value,  # pkl
                global_state['params']['seed'],  # w0_seed,
                global_state['params']['latent_space'] == 'w+',  # w_plus
                'const',
                global_state['params']['trunc_psi'],  # trunc_psi,
                global_state['params']['trunc_cutoff'],  # trunc_cutoff,
                None,
                global_state['params']['lr']  # lr,
            )
            renderer._render_drag_impl(global_state['generator_params'],
                                       is_drag=False)
            image_raw = global_state['generator_params'].image

            create_images(image_raw, global_state, update_original=True)

            # return global_state, image_raw, global_state["draws"][
            #     "image_with_mask"]
            return global_state, image_raw

        form_update_image_seed_btn.click(
            on_change_update_image_seed,
            inputs=[form_seed_number, global_state],
            outputs=[global_state, form_image_draw],
        )

        def on_click_latent_space(latent_space, global_state):
            # import ipdb
            # ipdb.set_trace()
            global_state['params']['latent_space'] = latent_space
            renderer = global_state['renderer']
            renderer.update_optim_space(latent_space == 'w+')
            return global_state

        form_latent_space.change(
            on_click_latent_space,
            inputs=[form_latent_space, global_state],
            outputs=[global_state]
        )

        # Tools tab listeners
        def on_change_dropdown_points(curr_point, global_state):
            global_state["curr_point"] = curr_point
            image_draw = draw_points_on_image(
                global_state["images"]["image_raw"],
                global_state["points"],
                global_state["curr_point"],
            )
            print(f'select: {curr_point}')
            return global_state, image_draw

        # ==== Params
        form_lambda_number.change(
            partial(on_change_single_global_state,
                    ["params", "motion_lambda"]),
            inputs=[form_lambda_number, global_state],
            outputs=[global_state],
        )

        def on_click_udate_lr(lr, global_state):

            global_state["params"]["lr"] = lr
            renderer = global_state['renderer']
            renderer.update_lr(lr)
            print('New optimizer: ')
            print(renderer.w_optim)

            return global_state

        update_lr_number.click(
            on_click_udate_lr,
            inputs=[form_lr_number, global_state],
            outputs=[global_state],
        )

        form_magnitude_direction_in_pixels_number.change(
            partial(
                on_change_single_global_state,
                ["params", "magnitude_direction_in_pixels"],
            ),
            inputs=[form_magnitude_direction_in_pixels_number, global_state],
            outputs=[global_state],
        )

        form_r1_in_pixels_number.change(
            partial(
                on_change_single_global_state,
                ["params", "r1_in_pixels"],
                map_transform=lambda x: int(x),
            ),
            inputs=[form_r1_in_pixels_number, global_state],
            outputs=[global_state],
        )

        form_r2_in_pixels_number.change(
            partial(
                on_change_single_global_state,
                ["params", "r2_in_pixels"],
                map_transform=lambda x: int(x),
            ),
            inputs=[form_r2_in_pixels_number, global_state],
            outputs=[global_state],
        )

        def on_click_start(global_state):
            p_in_pixels = []
            t_in_pixels = []
            valid_points = []

            # Prepare the points for the inference
            if len(global_state["points"]) == 0:

                image_draw = draw_points_on_image(
                    global_state["draws"]["image_with_points"],
                    global_state["points"],
                    global_state["curr_point"],
                )
                return global_state, 0, image_draw, image_draw, gr.File.update(
                    visible=False)

            # Transform the points into torch tensors
            for key_point, point in global_state["points"].items():
                try:
                    p_start = point.get("start_temp", point["start"])
                    p_end = point["target"]

                    if p_start is None or p_end is None:
                        continue

                except KeyError:
                    continue

                p_in_pixels.append(p_start)
                t_in_pixels.append(p_end)
                valid_points.append(key_point)

            mask_in_pixels = torch.tensor(
                global_state["images"]["image_mask"]).float()

            renderer: Renderer = global_state["renderer"]
            global_state['temporal_params']['stop'] = False
            step_idx = 0
            while True:
                if global_state["temporal_params"]["stop"]:
                    break

                # do drage here!
                renderer._render_drag_impl(
                    global_state['generator_params'],
                    p_in_pixels,  # point
                    t_in_pixels,  # target
                    mask_in_pixels,  #  mask,
                    global_state['params']['motion_lambda'],  # lambda_mask
                    reg=0,
                    feature_idx=5,  # NOTE: do not support change for now
                    r1=global_state['params']['r1_in_pixels'],  # r1
                    r2=global_state['params']['r2_in_pixels'],  # r2
                    # random_seed     = 0,
                    # noise_mode      = 'const',
                    trunc_psi=global_state['params']['trunc_psi'],
                    # force_fp32      = False,
                    # layer_name      = None,
                    # sel_channels    = 3,
                    # base_channel    = 0,
                    # img_scale_db    = 0,
                    # img_normalize   = False,
                    # untransform     = False,
                    is_drag=True)

                if step_idx % global_state['draw_interval'] == 0:
                    for key_point, p_i, t_i in zip(valid_points, p_in_pixels,
                                                   t_in_pixels):
                        # global_state["points"][key_point]["start_temp"] = p_i.tolist()
                        # global_state["points"][key_point]["target"] = t_i.tolist()
                        global_state["points"][key_point]["start_temp"] = p_i
                        global_state["points"][key_point]["target"] = t_i

                    create_images(global_state['generator_params']['image'],
                                  global_state)

                yield (
                    global_state,
                    step_idx,
                    # global_state["draws"]["image_with_points"],
                    # global_state["draws"]["image_with_mask"],
                    global_state["draws"]["image_with_points"],
                    gr.File.update(visible=False),
                )

                # increate step
                step_idx += 1

            image_result = global_state['generator_params']['image']
            # create_images(global_state['image'], global_state)
            create_images(global_state['generator_params']['image'],
                          global_state)

            fp = NamedTemporaryFile(suffix=".png", delete=False)
            image_result.save(fp, "PNG")

            yield (
                global_state,
                step_idx,
                # global_state["draws"]["image_with_points"],
                # global_state["draws"]["image_with_mask"],
                global_state["draws"]["image_with_points"],
                gr.File.update(visible=True, value=fp.name),
            )

        form_start_btn.click(
            on_click_start,
            inputs=[global_state],
            # outputs=[
            #     global_state, form_steps_number, form_image_draw,
            #     form_image_mask_draw, form_download_result_file
            # ],
            outputs=[
                global_state, form_steps_number, form_image_draw,
                form_download_result_file
            ],
        )

        def on_click_stop(global_state):
            global_state["temporal_params"]["stop"] = True

            return global_state

        form_stop_btn.click(on_click_stop,
                            inputs=[global_state],
                            outputs=[global_state])

        form_draw_interval_number.change(
            partial(
                on_change_single_global_state,
                "draw_interval",
                map_transform=lambda x: int(x),
            ),
            inputs=[form_draw_interval_number, global_state],
            outputs=[global_state],
        )

        def on_click_remove_point(global_state):
            choice = global_state["curr_point"]
            del global_state["points"][choice]

            choices = list(global_state["points"].keys())

            if len(choices) > 0:
                global_state["curr_point"] = choices[0]

            return (
                gr.Dropdown.update(choices=choices, value=choices[0]),
                global_state,
            )

        # Mask
        def on_click_reset_mask(global_state):
            # if global_state['edit_mask_mode'] == 'add_points':
            #     return global_state, global_state["draws"]["image_with_mask"]

            global_state["images"]["image_mask"] = np.ones(
                (
                    global_state["images"]["image_raw"].size[1],
                    global_state["images"]["image_raw"].size[0],
                ),
                dtype=np.uint8,
            )
            # global_state["draws"]["image_with_mask"] = draw_mask_on_image(
            #     global_state["images"]["image_raw"],
            #     global_state["images"]["image_mask"])
            # return global_state, global_state["draws"]["image_with_mask"]

            create_images(global_state['images']['image_raw'],
                          global_state, False)
            return global_state

        form_reset_mask_btn.click(
            on_click_reset_mask,
            inputs=[global_state],
            # outputs=[global_state, form_image_mask_draw],
            outputs=[global_state],
        )

        form_radius_mask_number.change(
            partial(
                on_change_single_global_state,
                "radius_mask",
                map_transform=lambda x: int(x),
            ),
            inputs=[form_radius_mask_number, global_state],
            outputs=[global_state],
        )

        # Image
        def on_click_enable_mask(global_state):
            global_state['edidint_state'] = 'add_mask'
            return (
                global_state,
                # gr.Image.update(visible=False),
                # gr.Image.update(visible=True),
                gr.Image.update(interactive=True),
            )

        enable_add_mask.click(
            on_click_enable_mask,
            inputs=[global_state],
            # outputs=[global_state, form_image_draw, form_image_mask_draw],
            outputs=[global_state, form_image_draw],
        )

        def on_click_enable_add_points(global_state):
            global_state['editing_state'] = 'add_points'
            return (
                global_state,
                gr.Image.update(interactive=False)
                # gr.Image.update(visible=True),
                # gr.Image.update(visible=False),
            )

        enable_add_points.click(
            on_click_enable_add_points,
            inputs=[global_state],
            # outputs=[global_state, form_image_draw, form_image_mask_draw],
            outputs=[global_state, form_image_draw],
        )

        def on_click_image(global_state, evt: gr.SelectData):
            """This function only support click for point selection
            """
            xy = evt.index
            if global_state['editing_state'] != 'add_points':
                print(f'In {global_state["editing_state"]} state. '
                      'Do not add points.')

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

            image_draw = draw_points_on_image(
                global_state["images"]["image_raw"],
                global_state["points"],
                None,  # NOTE: we hight light all points
                # global_state["curr_point"],
            )

            global_state["draws"]["image_with_points"] = image_draw

            return global_state, image_draw

        form_image_draw.select(
            on_click_image,
            inputs=[global_state],
            outputs=[global_state, form_image_draw],
        )

        def on_click_undo(global_state):

            points = global_state["points"]

            point_idx = get_latest_points_pair(points)
            if point_idx is None:
                print('No point has been added. Undo nothing.')
                pass
            elif points[point_idx].get('target', None) is None:
                removed_point = points.pop(point_idx)
                # points[point_idx]['target'] = None
                print(f'Undo start point: {removed_point}')
            else:
                removed_point = points[point_idx]['target']
                points[point_idx]['target'] = None
                print(f'Undo target point: {removed_point}')

            image_draw = draw_points_on_image(
                global_state["images"]["image_raw"],
                global_state["points"],
                None,  # NOTE: we hight light all points
                # global_state["curr_point"],
            )
            global_state["draws"]["image_with_points"] = image_draw

            return global_state, image_draw

        undo_points.click(on_click_undo,
                          inputs=[global_state],
                          outputs=[global_state, form_image_draw])

        # def on_click_mask(global_state, evt: gr.SelectData):
        #     xy = evt.index

        #     radius_mask = int(global_state["radius_mask"])

        #     image_mask = np.uint8(255 * global_state["images"]["image_mask"])
        #     image_mask = cv2.circle(image_mask, xy, radius_mask, 0, -1) > 127
        #     global_state["images"]["image_mask"] = image_mask

        #     image_with_mask = draw_mask_on_image(
        #         global_state["images"]["image_raw"], image_mask)
        #     global_state["draws"]["image_with_mask"] = image_with_mask

        #     return global_state, image_with_mask

        # form_image_mask_draw.select(
        #     on_click_mask,
        #     inputs=[global_state],
        #     outputs=[global_state, form_image_mask_draw],
        # )

gr.close_all()
app.queue(concurrency_count=5, max_size=20)
app.launch()

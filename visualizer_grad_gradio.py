from ipdb import iex
import os
import os.path as osp
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
cache_dir = './checkpoints'
valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(cache_dir, f) for f in os.listdir(cache_dir) if (f.endswith('pkl') and osp.exists(osp.join(cache_dir, f)))
}

init_pkl = 'stylegan_human_v2_512'


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def clear_state(global_state, target=None):
    """Clear target history state from global_state
    If target is not defined, points and mask will be both removed.
    1. set global_state['points'] as empty dict
    2. set global_state['mask'] as full-one mask.
    """
    if target is None:
        target = ['point', 'mask']
    if not isinstance(target, list):
        target = [target]
    if 'point' in target:
        global_state['points'] = dict()
        print('Clear Points State!')
    if 'mask' in target:
        image_raw = global_state["images"]["image_raw"]
        global_state['images']['image_mask'] = np.ones(
            (image_raw.size[1], image_raw.size[0]), dtype=np.uint8)
        print('Clear mask State!')

    return global_state


def create_images(image_raw, global_state, update_original=False):
    if isinstance(image_raw, torch.Tensor):
        image_raw = image_raw.cpu().numpy()
        image_raw = Image.fromarray(image_raw)

    if update_original:
        global_state["images"]["image_orig"] = image_raw.copy()
        print('Update image_orig to image_raw')

    global_state["images"]["image_raw"] = image_raw
    global_state["draws"]["image_with_points"] = draw_points_on_image(
        image_raw, global_state["points"], global_state["curr_point"])

    if 'image_mask' not in global_state['images']:
        global_state["images"]["image_mask"] = np.ones(
            (image_raw.size[1], image_raw.size[0]), dtype=np.uint8)
    else:
        # rebuild mask to avoid shape mismatch
        mask = global_state["images"]["image_mask"]
        if (mask.shape[0] != image_raw.size[1]
                or mask.shape[1] != image_raw.size[0]):
            global_state["images"]["image_mask"] = np.ones(
                (image_raw.size[1], image_raw.size[0]), dtype=np.uint8)

    global_state["draws"]["image_with_mask"] = draw_mask_on_image(
        global_state["images"]["image_raw"],
        global_state["images"]["image_mask"])


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
        "draw_interval": 1,
        "radius_mask": 51,
        "renderer": Renderer(),
        "points": {},
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',
        'pretrained_weight': init_pkl
    })

    # init image
    # import ipdb
    # ipdb.set_trace()
    global_state.value['renderer'].init_network(
        global_state.value['generator_params'],  # res
        valid_checkpoints_dict[global_state.value['pretrained_weight']],  # pkl
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

        with gr.Column(scale=3):
            # gr.HighlightedText([('Network & Latent')], show_label=False)

            # network and latent
            with gr.Row():

                with gr.Column(scale=1, min_width=10):
                    gr.Markdown(value='Pickle', show_label=False)

                with gr.Column(scale=4, min_width=10):
                    form_pretrained_dropdown = gr.Dropdown(
                        choices=valid_checkpoints_dict.keys(),
                        label="Pretrained model",
                        value=init_pkl,
                    )

            with gr.Row():
                with gr.Column(scale=1, min_width=10):
                    gr.Markdown(value='Latent', show_label=False)

                with gr.Column(scale=4, min_width=10):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            form_seed_number = gr.Number(
                                        value=42,
                                        interactive=True,
                                        label="Seed",
                                    )
                        with gr.Column(scale=2, min_width=10):
                            form_lr_number = gr.Number(
                                value=global_state.value["params"]["lr"],
                                interactive=True,
                                label="LR")

                    with gr.Row():
                        with gr.Column(scale=2, min_width=10):
                            form_reset_image = gr.Button("Reset image")
                        with gr.Column(scale=3, min_width=10):
                            form_latent_space = gr.Radio(
                                ['w', 'w+'],
                                value=global_state.value['params']['latent_space'],
                                interactive=True,
                                label='Latent space to optimize',
                                show_label=False,
                            )

            # drag
            with gr.Row():
                with gr.Column(scale=1, min_width=10):
                    gr.Markdown(value='Drag', show_label=False)
                with gr.Column(scale=4, min_width=10):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            enable_add_points = gr.Button('Add')
                        with gr.Column(scale=1, min_width=10):
                            undo_points = gr.Button('Reset')
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            form_start_btn = gr.Button("Start")
                        with gr.Column(scale=1, min_width=10):
                            form_stop_btn = gr.Button("Stop")

                    form_steps_number = gr.Number(
                        value=0, label="Steps",
                        interactive=False)

            with gr.Row():
                with gr.Column(scale=1, min_width=10):
                    gr.Markdown(value='Mask', show_label=False)
                with gr.Column(scale=4, min_width=10):
                    enable_add_mask = gr.Button('Edit')
                    # TODO: do not support yet
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            flex_area = gr.Button('Flexible Area')
                        with gr.Column(scale=1, min_width=10):
                            fixed_area = gr.Button('Fixed Area')
                    with gr.Row():
                        with gr.Column(scale=1, min_width=10):
                            form_reset_mask_btn = gr.Button("Reset mask")
                        with gr.Column(scale=1, min_width=10):
                            show_mask = gr.Checkbox(label='Show Mask', show_label=False) 

                    with gr.Row():
                        with gr.Column(scale=4, min_width=10):
                            form_radius_mask_number = gr.Number(
                                value=global_state.value["radius_mask"],
                                interactive=True,
                                label="Radius (pixels)")
                        with gr.Column(scale=1, min_width=10):
                            gr.Markdown(value='')
                            gr.Markdown('Radius')
                    with gr.Row():
                        with gr.Column(scale=4, min_width=10):
                            form_lambda_number = gr.Number(
                                value=global_state.value["params"][
                                    "motion_lambda"],
                                interactive=True,
                                label="Lambda",
                            )
                        with gr.Column(scale=1, min_width=10):
                            gr.Markdown(value='')
                            gr.Markdown('Lambda')

            # save
            with gr.Column(visible=False):
                form_download_result_file = gr.File(
                    label="Download result",
                    visible=False).style(full_width=True)

            # >>> some unused labels
            form_r1_in_pixels_number = gr.Number(
                value=global_state.value["params"]["r1_in_pixels"],
                interactive=True,
                label="R1 (pixels)", visible=False).style(full_width=False)
            form_r2_in_pixels_number = gr.Number(
                value=global_state.value["params"]["r2_in_pixels"],
                interactive=True,
                label="R2 (pixels)", visible=False).style(full_width=False)
            # <<< some unused labels

            form_draw_interval_number = gr.Number(
                value=global_state.value["draw_interval"],
                label="Draw Interval (steps)",
                interactive=True, visible=False)

        with gr.Column(scale=8):
            form_image_draw = gr.Image(
                global_state.value["draws"]["image_with_points"],
                elem_classes="image_nonselectable")
            form_image_mask_draw = gr.Image(
                global_state.value["draws"]["image_with_mask"],
                visible=False,
                elem_classes="image_nonselectable",
            )
            gr.Markdown(
                "Credits: Adrià Ciurana Lanau | info@dreamlearning.ai | OpenMMLab | ?"
            )


    # with gr.Row():
    #     # Left column
    #     with gr.Column(scale=0.7):

    #         with gr.Accordion("Network & latent"):

    #             with gr.Row():
    #                 with gr.Tab("Pretrained models"):
    #                     form_pretrained_dropdown = gr.Dropdown(
    #                         choices=valid_checkpoints_dict.keys(),
    #                         label="Pretrained model",
    #                         value=init_pkl,
    #                     )

    #             with gr.Row().style(equal_height=True):
    #                 with gr.Tab("Image seed"):
    #                     with gr.Row():
    #                         form_seed_number = gr.Number(
    #                             value=42,
    #                             interactive=True,
    #                             label="Seed",
    #                         )

    #                 with gr.Tab("Image projection"):
    #                     with gr.Row():
    #                         form_project_file = gr.File(
    #                             label="Image project file")
    #                         form_project_iterations_number = gr.Number(
    #                             value=1_000,
    #                             label="Image projection num steps",
    #                         )
    #                         form_update_image_project_btn = gr.Button(
    #                             "Run projection")

    #                 form_reset_image = gr.Button("Reset image")

    #     with gr.Accordion('Tools'):
    #         enable_add_points = gr.Button('Draw Points')
    #         # TODO: support mask
    #         # form_enable_add_mask = gr.Button('Draw Mask')
    #         undo_points = gr.Button('Reset Points')
    #         enable_add_mask = gr.Button('Draw Mask')

    #         # TODO: add a list to show all control points

    #         form_reset_mask_btn = gr.Button("Reset mask").style(
    #             full_width=True)
    #         form_radius_mask_number = gr.Number(
    #             value=global_state.value["radius_mask"],
    #             interactive=True,
    #             label="Radius (pixels)",
    #         ).style(full_width=False)

    #     with gr.Row():
    #         # with gr.Tab("Run"):
    #         with gr.Row():
    #             with gr.Column():
    #                 form_start_btn = gr.Button("Start").style(
    #                     full_width=True)
    #                 form_stop_btn = gr.Button("Stop").style(
    #                     full_width=True)
    #             form_steps_number = gr.Number(
    #                 value=0, label="Steps",
    #                 interactive=False).style(full_width=False)
    #             form_draw_interval_number = gr.Number(
    #                 value=global_state.value["draw_interval"],
    #                 label="Draw Interval (steps)",
    #                 interactive=True,
    #             ).style(full_width=False)
    #             form_download_result_file = gr.File(
    #                 label="Download result",
    #                 visible=False).style(full_width=True)

    #         # with gr.Tab("Hyperparameters"):
    #     with gr.Row():
    #         form_lambda_number = gr.Number(
    #             value=global_state.value["params"]["motion_lambda"],
    #             interactive=True,
    #             label="Lambda",
    #         ).style(full_width=True)
    #         with gr.Column():
    #             form_lr_number = gr.Number(
    #                 value=global_state.value["params"]["lr"],
    #                 interactive=True,
    #                 label="LR",
    #             ).style(full_width=True)
    #             update_lr_number = gr.Button('Update')
    #         form_magnitude_direction_in_pixels_number = gr.Number(
    #             value=global_state.value["params"]
    #             ["magnitude_direction_in_pixels"],
    #             interactive=True,
    #             label=("Magnitude direction of d vector"
    #                    " (pixels)"),
    #         ).style(full_width=True)

    #     # with gr.Row():
    #         form_r1_in_pixels_number = gr.Number(
    #             value=global_state.value["params"]["r1_in_pixels"],
    #             interactive=True,
    #             label="R1 (pixels)",
    #         ).style(full_width=False)
    #         form_r2_in_pixels_number = gr.Number(
    #             value=global_state.value["params"]["r2_in_pixels"],
    #             interactive=True,
    #             label="R2 (pixels)",
    #         ).style(full_width=False)

    #         form_latent_space = gr.Radio(
    #             ['w', 'w+'],
    #             value=global_state.value['params']['latent_space'],
    #             interactive=True,
    #             label='Latent space to optimize',
    #         )

    #     with gr.Column():
    #         form_image_draw = gr.Image(
    #             global_state.value["draws"]["image_with_points"],
    #             elem_classes="image_nonselectable")
    #         form_image_mask_draw = gr.Image(
    #             global_state.value["draws"]["image_with_mask"],
    #             visible=False,
    #             elem_classes="image_nonselectable",
    #         )
    #         gr.Markdown(
    #             "Credits: Adrià Ciurana Lanau | info@dreamlearning.ai | OpenMMLab | ?"
    #         )

        # Network & latents tab listeners
        def on_change_pretrained_dropdown(pretrained_value, global_state,
                                          seed):
            """Function to handle model change.
            1. download model if need (TODO:)
            2. re-init network with renderer.init_network
            3. clear all, including images, because the original mask may mismatch with the new checkpoint
            3. re-draw image with random seed
            """
            pretrained_pkl_path = valid_checkpoints_dict[pretrained_value]
            global_state['pretrained_weight'] = pretrained_value
            renderer: Renderer = global_state["renderer"]

            renderer.init_network(
                global_state['generator_params'],  # res
                pretrained_pkl_path,  # pkl
                global_state['params']['seed'],  # w0_seed,
                global_state['params']['latent_space'] == 'w+',  # w_plus
                'const',
                global_state['params']['trunc_psi'],  # trunc_psi,
                global_state['params']['trunc_cutoff'],  # trunc_cutoff,
                None,
                global_state['params']['lr']  # lr,
            )
            clear_state(global_state)

            renderer._render_drag_impl(global_state['generator_params'],
                                       is_drag=False)
            image_raw = global_state['generator_params'].image
            create_images(image_raw, global_state, update_original=True)

            return global_state, image_raw, global_state["draws"][
                "image_with_mask"]

        form_pretrained_dropdown.change(
            on_change_pretrained_dropdown,
            inputs=[form_pretrained_dropdown, global_state, form_seed_number],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        def on_click_reset_image(global_state):
            """Reset image to the original one and clear all states
            NOTE: this button will be disabled with optimization is running.
            1. generate image with the original random seed and clear optimizer & w states (renderer.init_network)
            2. clear mask and point state (clear_state)
            3. re-draw image (create_image)
            """
            renderer = global_state["renderer"]
            seed = int(global_state["params"]["seed"])
            renderer.init_network(
                global_state['generator_params'],  # res
                renderer.pkl,  # pkl
                # pretrained_value,  # pkl
                seed,  # w0_seed,
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

            clear_state(global_state)

            create_images(image_raw, global_state, update_original=True)

            return global_state, image_raw, global_state["draws"][
                "image_with_mask"]

        form_reset_image.click(
            on_click_reset_image,
            inputs=[global_state],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        # Update parameters
        def on_change_update_image_seed(seed, global_state):
            """Function to handle generation seed change.
            1. generate image with new random seed and clear optimizer & w states (renderer.init_network)
            2. clear mask and point state (clear_state)
            3. re-draw image (create_images)
            """
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

            clear_state(global_state)

            create_images(image_raw, global_state, update_original=True)

            return global_state, image_raw, global_state["draws"][
                "image_with_mask"]

        form_seed_number.change(
            on_change_update_image_seed,
            inputs=[form_seed_number, global_state],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        def on_click_latent_space(latent_space, global_state):
            """Function to reset latent space to optimize.
            NOTE: this function we reset the image and all controls
            0. re-generate image
            1. clear all state
            2. re-draw image
            3. update latent space
            """
            clear_state(global_state)

            renderer: Renderer = global_state["renderer"]
            renderer.init_network(
                global_state['generator_params'],  # res
                renderer.pkl,  # pkl
                global_state['params']['seed'],  # w0_seed,
                global_state['params']['latent_space'] == (latent_space == 'w+'),  # w_plus
                'const',
                global_state['params']['trunc_psi'],  # trunc_psi,
                global_state['params']['trunc_cutoff'],  # trunc_cutoff,
                None,
                global_state['params']['lr']  # lr,
            )
            renderer._render_drag_impl(global_state['generator_params'],
                                       is_drag=False)

            image_raw = global_state['generator_params'].image

            clear_state(global_state)
            create_images(image_raw, global_state, update_original=True)

            return global_state, image_raw, global_state["draws"][
                "image_with_mask"]

        form_latent_space.change(
            on_click_latent_space,
            inputs=[form_latent_space, global_state],
            outputs=[global_state, form_image_draw, form_image_mask_draw]
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

        def on_change_lr(lr, global_state):
            if lr == 0:
                print('lr is 0, do nothing.')
                return global_state
            else:
                global_state["params"]["lr"] = lr
                renderer = global_state['renderer']
                renderer.update_lr(lr)
                print('New optimizer: ')
                print(renderer.w_optim)
            return global_state 

        # def on_click_udate_lr(lr, global_state):

        #     global_state["params"]["lr"] = lr
        #     renderer = global_state['renderer']
        #     renderer.update_lr(lr)
        #     print('New optimizer: ')
        #     print(renderer.w_optim)

        #     return global_state

        # update_lr_number.click(
        #     on_click_udate_lr,
        #     inputs=[form_lr_number, global_state],
        #     outputs=[global_state],
        # )

        form_lr_number.change(
            on_change_lr, 
            inputs=[form_lr_number, global_state],
            outputs=[global_state],
        )

        # form_magnitude_direction_in_pixels_number.change(
        #     partial(
        #         on_change_single_global_state,
        #         ["params", "magnitude_direction_in_pixels"],
        #     ),
        #     inputs=[form_magnitude_direction_in_pixels_number, global_state],
        #     outputs=[global_state],
        # )

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
                # image_draw = draw_points_on_image(
                #     global_state["draws"]["image_with_points"],
                #     global_state["points"],
                #     global_state["curr_point"],
                # )
                image_draw = global_state['draws']['image_with_points']
                return (
                    global_state,
                    0,
                    image_draw,
                    image_draw,
                    gr.File.update(visible=False),
                    gr.Button.update(interactive=True),
                    gr.Button.update(interactive=True),
                    gr.Button.update(interactive=True),
                    gr.Button.update(interactive=True),
                    gr.Button.update(interactive=True),
                    # latent space
                    gr.Radio.update(interactive=True),
                    gr.Button.update(interactive=True),
                    # NOTE: disable stop button
                    gr.Button.update(interactive=False),
                )

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
            global_state['editing_state'] = 'running'

            # reverse points order
            p_to_opt = reverse_point_pairs(p_in_pixels)
            t_to_opt = reverse_point_pairs(t_in_pixels)
            step_idx = 0
            while True:
                if global_state["temporal_params"]["stop"]:
                    break

                # do drage here!
                renderer._render_drag_impl(
                    global_state['generator_params'],
                    p_to_opt,  # point
                    t_to_opt,  # target
                    mask_in_pixels,  # mask,
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
                    for key_point, p_i, t_i in zip(valid_points, p_to_opt,
                                                   t_to_opt):
                        # global_state["points"][key_point]["start_temp"] = p_i.tolist()
                        # global_state["points"][key_point]["target"] = t_i.tolist()
                        # global_state["points"][key_point]["start_temp"] = p_i
                        # global_state["points"][key_point]["target"] = t_i
                        global_state["points"][key_point]["start_temp"] = [
                            p_i[1], p_i[0]]
                        global_state["points"][key_point]["target"] = [
                            t_i[1], t_i[0]]

                    create_images(global_state['generator_params']['image'],
                                  global_state)

                yield (
                    global_state,
                    step_idx,
                    global_state["draws"]["image_with_points"],
                    global_state["draws"]["image_with_mask"],
                    gr.File.update(visible=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    # latent space
                    gr.Radio.update(interactive=False),
                    gr.Button.update(interactive=False),
                    # enable stop button in loop
                    gr.Button.update(interactive=True),
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
                0,  # reset step to 0 after stop.
                global_state["draws"]["image_with_points"],
                global_state["draws"]["image_with_mask"],
                gr.File.update(visible=True, value=fp.name),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                # latent space
                gr.Radio.update(interactive=True),
                gr.Button.update(interactive=True),
                # NOTE: disable stop button with loop finish
                gr.Button.update(interactive=False),
            )

        form_start_btn.click(
            on_click_start,
            inputs=[global_state],
            outputs=[
                global_state, form_steps_number, form_image_draw,
                form_image_mask_draw, form_download_result_file,
                # >>> buttons
                form_reset_image,
                enable_add_points,
                enable_add_mask,
                undo_points,
                form_reset_mask_btn,
                form_latent_space,
                form_start_btn,
                form_stop_btn,
                # <<< buttonm
            ],
        )

        def on_click_stop(global_state):
            """Function to handle stop button is clicked.
            1. send a stop signal by set global_state["temporal_params"]["stop"] as True
            2. Disable Stop button
            """
            global_state["temporal_params"]["stop"] = True

            return global_state, gr.Button.update(interactive=False)

        form_stop_btn.click(on_click_stop,
                            inputs=[global_state],
                            outputs=[global_state, form_stop_btn])

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
            global_state["draws"]["image_with_mask"] = draw_mask_on_image(
                global_state["images"]["image_raw"],
                global_state["images"]["image_mask"])
            return global_state, global_state["draws"]["image_with_mask"]

        form_reset_mask_btn.click(
            on_click_reset_mask,
            inputs=[global_state],
            outputs=[global_state, form_image_mask_draw],
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
                gr.Image.update(visible=False),
                gr.Image.update(visible=True),
            )

        enable_add_mask.click(
            on_click_enable_mask,
            inputs=[global_state],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        def on_click_enable_add_points(global_state):
            global_state['editing_state'] = 'add_points'
            return (
                global_state,
                gr.Image.update(visible=True),
                gr.Image.update(visible=False),
            )

        enable_add_points.click(
            on_click_enable_add_points,
            inputs=[global_state],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
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

        def on_click_clear_points(global_state):
            """Function to handle clear all control points
            1. clear global_state['points'] (clear_state)
            2. re-draw image
            """
            clear_state(global_state, target='point')
            create_images(global_state['images']['image_raw'], global_state)
            return global_state, global_state['draws']['image_with_points']

        undo_points.click(on_click_clear_points,
                          inputs=[global_state],
                          outputs=[global_state, form_image_draw])

        def on_click_mask(global_state, evt: gr.SelectData):
            xy = evt.index

            radius_mask = int(global_state["radius_mask"])

            image_mask = np.uint8(255 * global_state["images"]["image_mask"])
            image_mask = cv2.circle(image_mask, xy, radius_mask, 0, -1) > 127
            global_state["images"]["image_mask"] = image_mask

            image_with_mask = draw_mask_on_image(
                global_state["images"]["image_raw"], image_mask)
            global_state["draws"]["image_with_mask"] = image_with_mask

            return global_state, image_with_mask

        form_image_mask_draw.select(
            on_click_mask,
            inputs=[global_state],
            outputs=[global_state, form_image_mask_draw],
        )

gr.close_all()
app.queue(concurrency_count=5, max_size=20)
app.launch()

import gradio as gr

with gr.Blocks() as b:
    image = gr.Image(value='./ttt.png',
                     tool='sketch',
                     brush_radius=20,
                     interactive=True)
    disable_draw = gr.Button('Disable Draw')
    enable_draw = gr.Button('Enable Draw')

    def on_click_disable_draw(image: gr.Image):
        return gr.Image.update(value=image['image'], interactive=False)

    def on_click_enable_draw(image):
        return gr.Image.update(value=image['image'], interactive=True)

    disable_draw.click(on_click_disable_draw, inputs=[image], outputs=[image])
    enable_draw.click(on_click_enable_draw, inputs=[image], outputs=[image])
    # disable_draw.click(on_click_disable_draw, inputs=[image])
    # enable_draw.click(on_click_enable_draw, inputs=[image])

b.launch()

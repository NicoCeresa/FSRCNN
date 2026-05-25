import torch
import gradio as gr
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image

from fsrcnn import FSRCNN, FSRCNN_s

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATHS = {
    2: 'app/api/models/FSRCNN_2s_10e_1b_0.2.0.pth',
    3: 'app/api/models/FSRCNN_3s_10e_1b_0.2.0.pth',
    4: 'app/api/models/FSRCNN_4s_10e_1b_0.2.0.pth',
}

def load_model(scale: int) -> FSRCNN:
    model = FSRCNN(scale=scale).to(device)
    state_dict = torch.load(MODEL_PATHS[scale], map_location=device, weights_only=True)
    state_dict = FSRCNN.remap_legacy_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model

models = {scale: load_model(scale) for scale in MODEL_PATHS}

to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

SCALE_OPTIONS = ['2×', '3×', '4×']

def upscale(image: Image.Image, scale_label: str) -> Image.Image:
    scale = int(scale_label[0])
    tensor = to_tensor(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = models[scale](tensor)
    return to_pil_image(output.squeeze(0).clamp(0, 1).cpu())


with gr.Blocks(title='FSRCNN — Super Resolution') as demo:
    gr.Markdown('# FSRCNN — Super Resolution')
    gr.Markdown('Fast Super-Resolution CNN ([Dong et al., 2016](https://arxiv.org/abs/1608.00367)). Upload a low-resolution image and select an upscaling factor.')

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type='pil', label='Input Image', height=400, sources=['upload'])
            scale_radio = gr.Radio(choices=SCALE_OPTIONS, value='2×', label='Scale')
            run_btn = gr.Button('Upscale', variant='primary')
        with gr.Column():
            output_image = gr.Image(type='pil', label='Upscaled Output', height=400, show_download_button=True)

    run_btn.click(fn=upscale, inputs=[input_image, scale_radio], outputs=output_image)

    gr.Examples(
        examples=[['images/bee_OG.png', '2×'], ['images/china_og.png', '3×']],
        inputs=[input_image, scale_radio],
    )

if __name__ == '__main__':
    demo.launch()

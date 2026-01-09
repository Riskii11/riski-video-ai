import gradio as gr
import torch
from diffusers import AnimateDiffPipeline
from prompt_engine import build_prompt

pipe = AnimateDiffPipeline.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16
)
pipe.to("cuda")

def generate_video(subject, action, mood):
    prompt = build_prompt(subject, action, mood)
    video = pipe(
        prompt=prompt,
        num_frames=16,
        guidance_scale=7.5
    ).frames
    return video

ui = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Textbox(label="Subject"),
        gr.Textbox(label="Action"),
        gr.Dropdown(
            ["calm", "dramatic", "dark", "mysterious"],
            label="Mood"
        )
    ],
    outputs=gr.Video(),
    title="Riski Free Video AI",
    description="Free cinematic AI video generator (limited)"
)

ui.launch()

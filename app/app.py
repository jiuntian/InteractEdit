import gc
import json
import os
from PIL import Image
import gradio as gr
from huggingface_hub import model_info
from huggingface_hub.errors import HfHubHTTPError, HFValidationError
import torch

from pipeline import InteractEditPipeline
from misc import SegmentationValueError, get_output_dir, save_details

# # load for edit
# pipeline = InteractEditPipeline.load_trained_pipeline(
#     base_model = None,
#     finetune_path = None,
# )

base_model_list = ['SG161222/RealVisXL_V5.0',
                   'SG161222/RealVisXL_V4.0',
                   'stabilityai/stable-diffusion-xl-base-1.0',
                   'John6666/nova-anime-xl-ilv30happynewyear-sdxl',
                   'cagliostrolab/animagine-xl-4.0',  # cfg 6, steps 25, euler a
                   'John6666/hassaku-xl-hentai-v13-better-eyes-sdxl',
                   'John6666/hassaku-xl-illustrious-v12style-sdxl', # v1.3 not available yet
                   'John6666/prefect-pony-xl-v50-sdxl',
                   'John6666/nova-anime-xl-pony-v7happyhalloween-sdxl',
                   'John6666/nova-anime-xl-ilv40happyvalentine-sdxl',
                   ]


def verify_hf_model(model):
    try:
        # Attempt to fetch model info from Hugging Face
        model_info(model)
    except (HfHubHTTPError, HFValidationError) as e:
        # Raise an exception if the model does not exist
        raise gr.Error(
            f"Model '{model}' does not exist or is private.", duration=0) from e


def train(image, sbj, obj, action, detections, base_model, save_name,
          initial_learning_rate, learning_rate, text_encoder_lr,
          train_batch_size, rank, text_encoder_rank,
          lambda_attention, phase1_train_steps, phase2_train_steps,
          resolution,
          adam_beta1, adam_beta2,
          adam_weight_decay, adam_weight_decay_text_encoder,
          optimizer, use_8bit_adam,
          enable_xformers, seed, use_dora, detected_image,
          clip_skip, lora_type,
          progress=gr.Progress(track_tqdm=True)):

    if image is None:
        raise gr.Error("Source Image cannot be blank!", duration=0)
    if sbj is None or sbj == "":
        raise gr.Error(
            "Subject cannot be blank! Enter \"person\" instead.", duration=0)
    if obj is None or obj == "":
        raise gr.Error(
            "Object cannot be blank! Enter the interacting object.", duration=0)
    verify_hf_model(base_model)
    if detections is None:
        raise gr.Error("Must detect for object masks first!", duration=0)

    output_dir = get_output_dir(save_name)

    InteractEditPipeline.train(
        base_model=base_model,
        output_dir=output_dir,
        image=image,
        sbj=sbj,
        obj=obj,
        action=action,
        initializer_tokens=[sbj, obj, "background"],
        detections=detections,
        initial_learning_rate=initial_learning_rate,
        learning_rate=learning_rate,
        text_encoder_lr=text_encoder_lr,
        train_batch_size=train_batch_size,
        rank=rank,
        text_encoder_rank=text_encoder_rank,
        lambda_attention=lambda_attention,
        phase1_train_steps=phase1_train_steps,
        phase2_train_steps=phase2_train_steps,
        resolution=resolution,
        adam_beta1=adam_beta1, adam_beta2=adam_beta2,
        adam_weight_decay=adam_weight_decay,
        adam_weight_decay_text_encoder=adam_weight_decay_text_encoder,
        optimizer=optimizer,
        use_8bit_adam=use_8bit_adam,
        enable_xformers=enable_xformers,
        seed=seed,
        use_dora=use_dora,
        clip_skip=clip_skip,
        lora_type=lora_type,
    )

    save_details(image,
                 sbj, obj, action, detections, base_model, save_name,
                 initial_learning_rate, learning_rate, text_encoder_lr,
                 train_batch_size, rank, text_encoder_rank,
                 lambda_attention, phase1_train_steps, phase2_train_steps,
                 resolution,
                 adam_beta1, adam_beta2,
                 adam_weight_decay, adam_weight_decay_text_encoder,
                 optimizer, use_8bit_adam,
                 enable_xformers, seed, use_dora, detected_image, output_dir, clip_skip)
    
    clear_cache()

    return f"Done, saved in {output_dir}"


def detect_plot(image: Image.Image,
                sbj: str,
                obj: str,
                size: int,):

    if image is None:
        raise gr.Error("Source Image cannot be blank!", duration=0)
    if sbj is None or sbj == "":
        raise gr.Error(
            "Subject cannot be blank! Enter \"person\" instead.", duration=0)
    if obj is None or obj == "":
        raise gr.Error(
            "Object cannot be blank! Enter the interacting object.", duration=0)
    try:
        res = InteractEditPipeline.detect_plot(
            image=image, sbj=sbj, obj=obj, size=size)
    except SegmentationValueError as e:
        raise gr.Error(
            str(e) + " Please try another image or check with provided label.", duration=0)
        
    clear_cache()

    return res


def load_trained_list(default_output_dir="./outputs"):
    choices = []

    for dirpath, dirnames, filenames in os.walk(default_output_dir):
        if "inversion.json" in filenames:  # Check if the file exists in the current folder
            file_path = os.path.join(dirpath, "inversion.json")
            try:
                # Get the modification time of the file
                mod_time = os.path.getmtime(file_path)
            except OSError:
                # If there is an error, use a default value
                mod_time = 0
            # Append a tuple of (modification time, directory path)
            choices.append((mod_time, dirpath))
    
    # Sort the list by modification time in descending order (latest first)
    choices.sort(key=lambda x: x[0], reverse=True)
    
    # Extract just the directory paths, now sorted
    sorted_dirs = [dirpath for _, dirpath in choices]


    return gr.Dropdown(choices=sorted_dirs, label="Inverted Weight",
                       scale=10, interactive=True)


def update_trained_model(selected_model):
    json_path = os.path.join(selected_model, "inversion.json")
    if not os.path.exists(json_path):
        raise gr.Error(f"{selected_model} not found!", duration=0)
    info = json.load(open(json_path))
    subject = info['subject']
    object = info['object']
    edit_base_model = info['base_model']
    
    pipeline, _ = update_pipeline(edit_base_model, selected_model)

    return subject, object, edit_base_model, pipeline

def update_pipeline(edit_base_model, selected_model,
                    progress=gr.Progress(track_tqdm=True)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = InteractEditPipeline.load_trained_pipeline(
        base_model = edit_base_model,
        finetune_path = selected_model,
    ).to(device, dtype=torch.float16)
    
    return pipeline, edit_base_model


def edit(pipeline: InteractEditPipeline, target_action,
         num_images_per_prompt, inference_steps,
         guidance_scale, seed, edit_clip_skip, 
         negative_prompt, denoising_end,
         progress=gr.Progress(track_tqdm=True)):
    if pipeline is None:
        raise gr.Error("Pipeline is not loaded. Select a inverted weight first.", duration=0)
    if target_action is None or target_action == "":
        raise gr.Error("Target action cannot be blank.", duration=0)
    
    prompt = f"a photo of <asset0> {target_action} <asset1> at <asset2>"
    
    out = pipeline(prompt=prompt,
                   num_images_per_prompt=num_images_per_prompt,
                   num_inference_steps=inference_steps,
                   guidance_scale=guidance_scale,
                   generator=torch.Generator(device=pipeline.device).manual_seed(seed),
                   clip_skip=edit_clip_skip,
                   negative_prompt=negative_prompt,
                   denoising_end=denoising_end,
                   )
    return out.images

def clear_cache(element=None):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


with gr.Blocks() as demo:
    gr.Markdown("Invert source image first or load inverted weight for editing.")
    with gr.Tab("Inversion"):
        with gr.Row():
            with gr.Column(scale=1):  # left
                source_image = gr.Image(label="Source Image", type="pil")
                with gr.Group():
                    with gr.Row():
                        subject = gr.Textbox(label="Subject", value="person")
                        object = gr.Textbox(label="Object")
                    action = gr.Textbox(label="Action (optional)", visible=False)
            with gr.Column(scale=1):
                detected_image = gr.Plot(
                    label="Detected Mask", format="png", container=False)
                detect_button = gr.Button("Detect Masks")
            with gr.Column(scale=1):  # right
                base_model = gr.Dropdown(label="Base Model",
                                         allow_custom_value=True,
                                         interactive=True,
                                         choices=base_model_list)
                base_model.change(verify_hf_model, inputs=base_model)
                save_name = gr.Textbox(label="Save Name", value="trained")

                with gr.Row():
                    train_button = gr.Button("Train", variant="primary", scale=1)
                    train_cancel = gr.Button("Cancel", variant="stop", scale=0)
                train_result = gr.Text(label="Train Progress")

                with gr.Accordion("More Parameters", open=False):
                    with gr.Row():
                        initial_learning_rate = gr.Number(
                            label="Stage 1 Learning Rate", value=3e-4)
                        learning_rate = gr.Number(
                            label="Stage 2 Learning Rate", value=1e-4)

                    clip_skip = gr.Number(label="Clip Skip", value=0,)
                    
                    with gr.Row():
                        text_encoder_lr = gr.Number(
                            label="Stage 2 Text Encoder Learning Rate", value=5e-6, scale=4)
                        train_batch_size = gr.Number(
                            label="Batch Size", value=1, scale=1)
                    with gr.Row():
                        rank = gr.Number(label="Rank", value=64)
                        text_encoder_rank = gr.Number(
                            label="Text Encoder Rank", value=4)
                    with gr.Row():
                        phase1_train_steps = gr.Number(
                            label="Stage 1 Train Steps", value=1000)
                        phase2_train_steps = gr.Number(
                            label="Stage 2 Train Steps", value=200)
                    with gr.Row():
                        resolution = gr.Number(label="Resolution", value=512)
                        seed = gr.Number(label="Seed", value=-1)

                        enable_xformers = gr.Checkbox(
                            value=True, label="Use xformers")
                        use_dora = gr.Checkbox(value=False, label="Use DoRA")
                    lambda_attention = gr.Number(
                        label="Cross-Attn Loss weight", value=0.01)

                    with gr.Accordion("Optimizer Parameters", open=False):
                        lora_type = gr.Dropdown(choices=["LoRA", "SeRA"],)
                        adam_beta1 = gr.Number(label="Adam beta 1", value=0.9)
                        adam_beta2 = gr.Number(
                            label="Adam beta 2", value=0.999)
                        adam_weight_decay = gr.Number(
                            label="Adam Weight Decay", value=1e-04)
                        adam_weight_decay_text_encoder = gr.Number(
                            label="Adam Weight Decay for Text Encoder", value=1e-03)
                        optimizer = gr.Dropdown(
                            choices=['AdamW'], value="AdamW", label="Optimizer")
                        use_8bit_adam = gr.Checkbox(
                            value=False, label="Use 8-bit Adam")

        detections = gr.State()
        detect_button.click(detect_plot,
                            inputs=[source_image, subject, object, resolution],
                            outputs=[detections, detected_image])

        train_event = train_button.click(train,
                           inputs=[source_image, subject, object, action, detections,
                                   base_model, save_name,
                                   initial_learning_rate, learning_rate, text_encoder_lr,
                                   train_batch_size, rank, text_encoder_rank,
                                   lambda_attention, phase1_train_steps, phase2_train_steps,
                                   resolution,
                                   adam_beta1, adam_beta2,
                                   adam_weight_decay, adam_weight_decay_text_encoder,
                                   optimizer, use_8bit_adam,
                                   enable_xformers, seed, use_dora, detected_image, clip_skip, lora_type],
                           outputs=[train_result],)
        train_cancel.click(fn=None, cancels=[train_event], )

    with gr.Tab("Editing"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    reload_button = gr.Button(value="Load Weights\n🔄", scale=0)
                    trained_model = gr.Dropdown(
                choices=[], label="Inverted Weight", scale=1, )
                edit_base_model = gr.Dropdown(label="Base Model",
                                              allow_custom_value=True,
                                              interactive=True,
                                              choices=base_model_list)
                with gr.Group():
                    with gr.Row():
                        source_subject = gr.Textbox(label="Source Subject")
                        source_object = gr.Textbox(label="Source Object")
                    target_action = gr.Textbox(label="Target Action")
                    with gr.Accordion(label="More Parameters"):
                        negative_prompt = gr.Textbox(label="Negative Prompt")
                        with gr.Row():
                            edit_seed = gr.Number(label="Seed", value=0)
                            denoising_end = gr.Slider(
                                minimum=0., maximum=1.,
                                value=1., step=0.1, label="Denoising Strength",
                            )
                        with gr.Row():
                            num_images_per_prompt = gr.Number(label="Batch Size", value=4)
                            guidance_scale = gr.Number(label="Guidance Scale", value=5.0, step=0.5)
                            inference_steps = gr.Number(label="Inference Steps", value=50)
                            edit_clip_skip = gr.Number(label="Clip Skip", value=0)
                edit_button = gr.Button("Edit!", variant="primary")
            with gr.Column(scale=1):
                image_output = gr.Gallery(label="Edited Images")
        pipeline = gr.State(delete_callback=clear_cache)
        
        timer = gr.Timer(30,)
        timer.tick(clear_cache)
        
        reload_button.click(load_trained_list, outputs=[trained_model])
        edit_base_model.change(update_pipeline,
                               inputs=[edit_base_model, trained_model,],
                               outputs=[pipeline, edit_base_model])
        trained_model.change(update_trained_model, inputs=[trained_model], outputs=[
                             source_subject, source_object, edit_base_model, pipeline])
        edit_button.click(edit,
                          inputs=[pipeline, target_action,
                                  num_images_per_prompt, inference_steps,
                                  guidance_scale, edit_seed, edit_clip_skip,
                                  negative_prompt, denoising_end],
                          outputs=[image_output])

if __name__ == "__main__":
    demo.launch()

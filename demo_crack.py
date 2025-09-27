# FramePack-F1: Image-to-Video Diffusion Model Demo
# This script implements a video generation system that takes a static image and generates
# a video sequence based on a text prompt using the HunyuanVideo diffusion model.

# Import HuggingFace login helper for accessing gated models
from diffusers_helper.hf_login import login

import os

# Set HuggingFace cache directory to a local folder to avoid cluttering system cache
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# Core libraries for the video generation pipeline
import gradio as gr  # Web UI framework
import torch  # PyTorch for deep learning operations
import traceback  # Error handling
import einops  # Tensor reshaping operations
import safetensors.torch as sf  # Safe tensor loading
import numpy as np  # Numerical operations
import argparse  # Command line argument parsing
import math  # Mathematical operations

from PIL import Image  # Image processing
# HunyuanVideo specific components
from diffusers import AutoencoderKLHunyuanVideo  # VAE for encoding/decoding images to latents
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer  # Text encoders
# Custom helper functions for the HunyuanVideo pipeline
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
# The main transformer model for video generation
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
# Sampling pipeline for the diffusion process
from diffusers_helper.pipelines.k_diffusion_crack import sample_crack
# Memory management utilities for handling large models on limited VRAM
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
# Asynchronous processing for real-time UI updates
from diffusers_helper.thread_utils import AsyncStream, async_run
# UI progress bar components
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
# CLIP vision model for image understanding
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
# Utility for finding optimal image dimensions
from diffusers_helper.bucket_tools import find_nearest_bucket
# Video to numpy
from video_helper import video_to_numpy

# Command line argument parser for server configuration
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')  # Enable public sharing via gradio
parser.add_argument("--server", type=str, default='0.0.0.0')  # Server host address
parser.add_argument("--port", type=int, required=False)  # Server port
parser.add_argument("--inbrowser", action='store_true')  # Auto-open browser
args = parser.parse_args()

# Server configuration notes:
# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

# Detect available GPU memory and set high-VRAM mode accordingly
# High-VRAM mode (>60GB) loads all models to GPU simultaneously for faster inference
# Low-VRAM mode uses dynamic model swapping to fit in limited memory
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# Load the text encoders - these convert text prompts into embeddings
# LlamaModel: Primary text encoder for understanding the main prompt
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
# CLIPTextModel: Secondary text encoder for additional text understanding
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
# Tokenizers convert text to tokens that the encoders can process
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')

# VAE (Variational Autoencoder) - converts between pixel space and latent space
# This is crucial for efficiency as diffusion happens in compressed latent space
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

# CLIP Vision components for understanding the input image
# These help the model understand what's in the input image to guide video generation
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

# The main transformer model that generates video latents
# This is the core of the video generation system
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()

# Set all models to evaluation mode (no training, only inference)
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

# Enable memory optimizations for low-VRAM systems
if not high_vram:
    vae.enable_slicing()  # Process images in slices to reduce memory usage
    vae.enable_tiling()   # Process images in tiles to reduce memory usage

# Enable high-quality output mode for the transformer
# This uses FP32 precision for the final output layer for better quality
transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# Set appropriate data types for each model component
# Different models use different precisions for optimal performance/quality balance
transformer.to(dtype=torch.bfloat16)  # Main model uses bfloat16
vae.to(dtype=torch.float16)           # VAE uses float16
image_encoder.to(dtype=torch.float16) # Vision encoder uses float16
text_encoder.to(dtype=torch.float16)  # Text encoders use float16
text_encoder_2.to(dtype=torch.float16)

# Disable gradient computation for all models (inference only)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

# Memory management strategy based on available VRAM
if not high_vram:
    # Low-VRAM mode: Use dynamic model swapping
    # Models are loaded to GPU only when needed, then offloaded to save memory
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    # High-VRAM mode: Load all models to GPU for maximum speed
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

# Create async stream for real-time communication between worker and UI
stream = AsyncStream()

# Create output directory for generated videos
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()  # Disable gradient computation for inference
def worker(
    input_image: np.ndarray,
    input_video: np.ndarray | None,
    prompt: str, 
    n_prompt: str, 
    seed: int, 
    total_second_length: float, 
    latent_window_size: int, 
    steps: int, 
    cfg: float, 
    gs: float, 
    rs: float, 
    gpu_memory_preservation: float, 
    use_teacache: bool, 
    mp4_crf: int):
    """
    Main worker function that generates video from input image and text prompt.
    
    Args:
        input_image: Input image as numpy array
        prompt: Text description of desired video
        n_prompt: Negative prompt (what to avoid)
        seed: Random seed for reproducibility
        total_second_length: Desired video length in seconds
        latent_window_size: Number of latent frames to generate per section
        steps: Number of diffusion sampling steps
        cfg: Classifier-free guidance scale
        gs: Distilled guidance scale
        rs: Guidance rescale factor
        gpu_memory_preservation: GB of GPU memory to preserve
        use_teacache: Whether to use TeaCache optimization
        mp4_crf: MP4 compression quality (lower = better quality)
    """
    
    # Calculate how many sections we need to generate for the desired video length
    # Each section generates latent_window_size * 4 - 3 frames (due to overlap)
    # At 30 FPS, we need total_second_length * 30 frames total
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # Generate unique job ID for this generation
    job_id = generate_timestamp()

    # Initialize progress tracking
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU memory if in low-VRAM mode
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # === TEXT ENCODING PHASE ===
        # Convert text prompts into embeddings that the model can understand

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        # Load text encoders to GPU if in low-VRAM mode
        if not high_vram:
            # Since we only encode one text - that is one model move and one encode, 
            # offload is same time consumption since it is also one load and one encode.
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        # Encode the main prompt using both text encoders
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        # Handle negative prompt (what we want to avoid in the video)
        if cfg == 1:
            # If CFG is 1, we don't need negative embeddings (no guidance)
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            # Encode negative prompt for classifier-free guidance
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        # Crop or pad text embeddings to fixed length (512 tokens)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # === IMAGE PROCESSING PHASE ===
        # Prepare the input image for the model

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        # Get image dimensions and find optimal bucket size
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)  # Find closest supported resolution
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        # Save processed input image for reference
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        # Convert image to tensor and normalize to [-1, 1] range
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]  # Add batch and time dimensions

        print('IMAGE PT: ', input_image_pt.shape)

        # === VAE ENCODING PHASE ===
        # Convert input image to latent space for efficient processing

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        # Load VAE to GPU if in low-VRAM mode
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        # Encode input image to latent space
        start_latent = vae_encode(input_image_pt, vae)
        print('IMAGE LATENT: ', start_latent.shape)

        # do the same for the video
        input_video_np, input_video_pt, video_latent = None, None, None
        if input_video is not None:
            VF, VH, VW, VC = input_video.shape
            video_height, video_width = find_nearest_bucket(VH, VW, resolution=640)  # Find closest supported resolution
            input_video_np = np.stack([resize_and_center_crop(f, target_width=video_width, target_height=video_height) for f in input_video], axis=0)

            input_video_pt = torch.from_numpy(input_video_np).float() / 127.5 - 1
            input_video_pt = input_video_pt.permute(3, 0, 1, 2)[None, :]
            print('VIDEO PT: ', input_video_pt.shape)

            video_latent = vae_encode(input_video_pt, vae)
            print('VIDEO LATENT: ', video_latent.shape)


        # === CLIP VISION ENCODING PHASE ===
        # Extract visual features from input image to guide video generation

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        # Load image encoder to GPU if in low-VRAM mode
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        # Extract visual features from input image
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # === DATA TYPE CONVERSION ===
        # Ensure all embeddings are in the correct data type for the transformer

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # === DIFFUSION SAMPLING PHASE ===
        # Generate video latents using the diffusion process

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        # Set up random number generator for reproducible results
        rnd = torch.Generator("cpu").manual_seed(seed)

        # Initialize history buffers for temporal consistency
        # The model uses previous frames to maintain temporal coherence
        # Initially set to all zeros
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None

        print('HISTORY LATENTS: ', history_latents.shape)

        # Add the starting latent to history
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        # Generate video in sections to handle long sequences
        for section_index in range(total_latent_sections):
            # Check if user wants to stop generation
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')

            # Load transformer to GPU if in low-VRAM mode
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            # Configure TeaCache optimization if enabled
            # TeaCache speeds up inference but may slightly reduce quality for hands/fingers
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                """
                Callback function called during each diffusion step.
                Updates the UI with preview images and progress information.
                """
                # Get the current denoised latent
                preview = d['denoised']
                # Decode to image for preview (using fake decode for speed)
                preview = vae_decode_fake(preview)

                # Convert to displayable format
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')  # Rearrange for display

                # Check if user wants to stop
                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                # Calculate and display progress
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            # Prepare indices for different temporal scales
            # The model uses multiple temporal resolutions for better video quality
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            # Extract clean latents from history for different temporal scales
            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            frames = latent_window_size * 4 - 3
            print('FRAMES: ', frames)

            # Run the diffusion sampling process
            generated_latents = sample_crack(
                transformer=transformer,           # Main video generation model
                sampler='unipc',                   # Sampling algorithm (UniPC)
                width=width,                       # Video width
                height=height,                     # Video height
                frames=frames, # Number of frames to generate
                real_guidance_scale=cfg,           # Classifier-free guidance scale
                distilled_guidance_scale=gs,       # Distilled guidance scale
                guidance_rescale=rs,               # Guidance rescale factor
                num_inference_steps=steps,         # Number of diffusion steps
                generator=rnd,                     # Random number generator
                # Text embeddings
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                # Negative text embeddings
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,                        # Device to run on
                dtype=torch.bfloat16,             # Data type
                image_embeddings=image_encoder_last_hidden_state,  # Visual features from input image
                # Temporal consistency latents
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,                 # Progress callback
            )

            print('GENERATED LATENTS: ', generated_latents.shape)

            # Update history with newly generated latents
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            print(section_index, 'HISTORY LATENT: ', history_latents.shape)

            # === VAE DECODING PHASE ===
            # Convert generated latents back to pixel space

            if not high_vram:
                # Offload transformer and load VAE for decoding
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            # Get the current history of latents
            # This is all history with the original zeros removed
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            print('REAL HISTORY LATENTS: ', real_history_latents.shape)

            # Decode latents to pixels
            if history_pixels is None:
                # First section: decode all latents
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                # Subsequent sections: decode only new frames and blend with overlap
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                # Soft append with overlap to maintain temporal smoothness
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            print('HISTORY PIXELS: ', history_pixels.shape)

            # Clean up GPU memory
            if not high_vram:
                unload_complete_models()

            # Save current video progress
            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            # Notify UI of new video file
            stream.output_queue.push(('file', output_filename))
            
    except:
        # Handle any errors that occur during generation
        traceback.print_exc()

        # Clean up GPU memory on error
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    # Signal completion
    stream.output_queue.push(('end', None))
    return


def process(input_image, input_video_path, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    """
    Main processing function called by the Gradio UI.
    Manages the async worker and handles UI updates.
    """
    print('INPUT VIDEO: ', input_video_path)
    input_video = video_to_numpy(input_video_path) if input_video_path is not None else None
    if input_video is not None:
        print('INPUT VIDEO: ', input_video.shape)
    print('INPUT IMAGE: ', input_image.shape)

    global stream
    assert input_image is not None, 'No input image!'

    # Disable start button and enable end button
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    # Create new async stream for this generation
    stream = AsyncStream()

    # Start the worker function asynchronously
    async_run(worker, input_image, input_video, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf)

    output_filename = None

    # Main UI update loop
    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            # New video file generated - update the video display
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            # Progress update - show preview and progress bar
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            # Generation complete - re-enable start button
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    """
    Function to stop the current generation process.
    """
    stream.input_queue.push('end')


if __name__ == '__main__':
    # Predefined prompt examples for quick testing
    quick_prompts = [
        'The girl dances gracefully, with clear movements, full of charm.',
        'A character doing some simple body movements.',
    ]
    quick_prompts = [[x] for x in quick_prompts]  # Format for Gradio Dataset


    # === GRADIO UI SETUP ===
    # Create the web interface for the video generation system

    css = make_progress_bar_css()  # Custom CSS for progress bars
    block = gr.Blocks(css=css).queue()  # Create Gradio interface with queue for async processing
    with block:
        gr.Markdown('# Crack')  # Title
        with gr.Row():
            with gr.Column():
                # Input image upload
                input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
                # Input video upload
                input_video = gr.Video(sources='upload', format='mp4', label="Video", height=320)
                # Text prompt input
                prompt = gr.Textbox(label="Prompt", value='')
                # Quick prompt examples
                example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

                with gr.Row():
                    # Control buttons
                    start_button = gr.Button(value="Start Generation")
                    end_button = gr.Button(value="End Generation", interactive=False)

                with gr.Group():
                    # Generation parameters
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                    n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used in this implementation
                    seed = gr.Number(label="Seed", value=31337, precision=0)  # Random seed for reproducibility

                    # Video length control
                    total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                    latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change

                    # Sampling parameters
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                    # Guidance parameters
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                    rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                    # Memory management
                    gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                    # Output quality
                    mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")

            with gr.Column():
                # Output displays
                preview_image = gr.Image(label="Next Latents", height=200, visible=False)  # Real-time preview during generation
                result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)  # Final video output
                progress_desc = gr.Markdown('', elem_classes='no-generating-animation')  # Progress description
                progress_bar = gr.HTML('', elem_classes='no-generating-animation')  # Progress bar

        # Social media link
        gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

        # Connect UI elements to functions
        ips = [input_image, input_video, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf]
        start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
        end_button.click(fn=end_process)

    # Launch the web interface
    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )
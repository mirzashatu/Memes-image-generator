# Necessary libraries
import torch
import diffusers
import streamlit as st
import io

# Set the device and `dtype` for GPUs
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# The dictionary mapping style names to style strings
style_dict = {
    "none": "",
    "anime": "cartoon, animated, Studio Ghibli style, cute, Japanese animation",
    # A photograph on film suggests an artistic approach
    "photo": "photograph, film, 35 mm camera",
    "video game": "rendered in unreal engine, hyper-realistic, volumetric lighting, --ar 9:16 --hd --q 2",
    "watercolor": "painting, watercolors, pastel, composition",
}

# Load Stable Diffusion with caching
@st.cache_resource
def load_model():
    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device in ["cuda", "mps"] else None
    )
    pipeline.to(device)
    return pipeline


# The generate_images function
def generate_images(prompt, pipeline, n, guidance=7.5, steps=50, style="none"):
    style_text = style_dict[style]
    output = pipeline(
        [prompt + style_text] * n, guidance_scale=guidance, num_inference_steps=steps
    )
    return output.images

# The main function
def main():
    st.title("Stable Diffusion GUI")

    # Sidebar controls
    num_images = st.sidebar.number_input("Number of Images", min_value=1, max_value=10, value=1)
    prompt = st.sidebar.text_area("Text-to-Image Prompt")

    guidance_help = "Lower values follow the prompt less strictly. Higher values risk distorted images."
    guidance = st.sidebar.slider("Guidance", 2.0, 15.0, 7.5, help=guidance_help)

    steps_help = "More steps produces better images but takes longer."
    steps = st.sidebar.slider("Steps", 10, 150, 50, help=steps_help)

    style = st.sidebar.selectbox("Style", options=style_dict.keys())

    generate = st.sidebar.button("Generate Images")
    
    # Pre-load model in session state for faster subsequent generations
    if "pipeline" not in st.session_state:
        with st.spinner("Loading model... (this will take a minute for first time)"):
            st.session_state.pipeline = load_model()

    if generate:
        if not prompt.strip():
            st.sidebar.error("Please enter a prompt!")
            return
            
        with st.spinner("Generating images..."):
            images = generate_images(
                prompt, 
                st.session_state.pipeline,  # Use cached model
                num_images, 
                guidance, 
                steps, 
                style
            )
            
        # Display images with download buttons
        if images:
            cols = st.columns(2)
            for idx, im in enumerate(images):
                with cols[idx % 2]:
                    st.image(im, caption=f"Generated Image {idx+1}")
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Download",
                        data=byte_im,
                        file_name=f"generated_image_{idx+1}.png",
                        mime="image/png",
                    )
            

if __name__ == "__main__":
    main()
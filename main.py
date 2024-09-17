import streamlit as st
import av
import torch
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

# Initialize Streamlit app
st.title("Video Summarization App")

# Upload video file
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Specify the path to your locally downloaded model
    local_model_path = "llava"

    # Load the model from the local directory
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        local_model_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=False, 
    ).to(0)

    # Load the processor from the local directory
    processor = LlavaNextVideoProcessor.from_pretrained(local_model_path)

    def read_video_pyav(container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this video about"},
                {"type": "video"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Save the uploaded video to a temporary file
    video_path = f"temp_video.{uploaded_file.name.split('.')[-1]}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    container = av.open(video_path)

    # Sample uniformly 8 frames from the video
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)

    inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

    # Generate summary
    output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False )
    decoded_output = processor.decode(output[0][2:], skip_special_tokens=True)

    # Remove unwanted prefixes from the decoded text
    if "ASSISTANT:" in decoded_output:
        summary = decoded_output.split("ASSISTANT:")[1].strip()
    else:
        summary = decoded_output

    # Display the summary
    st.subheader("Video Summary:")
    st.write(summary)

    # Optionally display the video
    st.video(uploaded_file)

else:
    st.info("Please upload a video file to get a summary.")

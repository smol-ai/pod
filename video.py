import os
import json
import requests
from datetime import datetime
import uuid
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.all import resize
from PIL import Image
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def log(category, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{category.upper()}] {message}"
    print(log_message)
    with open("video_generation.log", "a") as log_file:
        log_file.write(log_message + "\n")

def generate_default_image(temp_folder: str, prompt="A modern and sleek AI-themed image for AI News Pod cover art.") -> str:
    log("IMAGE_GEN", "Generating default image with DALL·E...")
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    image_url = response.data[0].url
    log("IMAGE_GEN", f"Image generated: {image_url}")
    
    # Download and save the image
    image_response = requests.get(image_url)
    image_path = os.path.join(temp_folder, "default_image.png")
    with open(image_path, "wb") as img_file:
        img_file.write(image_response.content)
    log("IMAGE_GEN", f"Image saved to: {image_path}")
    return image_path

def resize_image(image, newsize):
    return image.resize(newsize, Image.LANCZOS)

def create_text_clip(segment, video_width, video_height):
    return TextClip(
        segment['text'],
        fontsize=segment.get('font_size', 48),
        font='Arial-Bold',
        color='white',
        bg_color='black',
        stroke_color='black',
        stroke_width=2,
        method='caption',
        size=(video_width * 0.9, None)  # Dynamic height based on text
    ).set_position(segment.get('position', ('center', video_height - video_height // 3))) \
     .set_start(segment['start_time']) \
     .set_duration(segment['duration'])

def create_video(audio_file: str, image_path: str, transcript_path: str) -> str:
    log("VIDEO_GEN", "Creating video from audio and image...")
    
    # Load audio and image
    audio = AudioFileClip(audio_file)
    image = ImageClip(image_path).set_duration(audio.duration)
    
    # Set a fixed fps for the video
    fps = 24
    
    # Resize the image
    image_resized = resize(image, height=1080).set_position("center")
    
    # Generate captions
    log("CAPTION_GEN", "Generating captions from transcript...")
    with open(transcript_path, "r") as f:
        transcript = json.load(f)
    
    # Calculate dimensions for the lower third
    video_width, video_height = image_resized.size
    
    # Generate text clips in parallel
    with ThreadPoolExecutor() as executor:
        text_clips = list(executor.map(
            lambda segment: create_text_clip(
                segment,
                video_width,
                video_height
            ),
            transcript
        ))
    
    # Combine all text clips into a single captions layer
    captions = CompositeVideoClip(text_clips, size=image_resized.size).set_duration(audio.duration)
    
    # Combine image and captions
    video = CompositeVideoClip([image_resized, captions]).set_audio(audio)
    
    # Export video with optimized settings
    video_path = "combined_video.mp4"
    video.write_videofile(
        video_path, 
        fps=fps, 
        codec="libx264", 
        audio_codec="aac",
        threads=4,
        preset="ultrafast",
        ffmpeg_params=["-crf", "28"]
    )
    log("VIDEO_GEN", f"Video created: {video_path}")
    return video_path

def main():
    log("PROCESS_START", "Starting the video generation process...")
    
    # Check if required files exist
    if not os.path.exists("combined_dialogue.mp3") or not os.path.exists("dialogue_transcript.json"):
        log("ERROR", "Required files (combined_dialogue.mp3 or dialogue_transcript.json) not found. Run main.py first.")
        return
    
    temp_folder = f"video_{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{uuid.uuid4()}"
    os.makedirs(temp_folder, exist_ok=True)
    log("TEMP_FOLDER", f"Created temporary folder: {temp_folder}")

    # Generate an image prompt based on the transcript
    with open("dialogue_transcript.json", "r") as f:
        transcript = json.load(f)
    
    # Use GPT-4o-mini to generate an image prompt
    prompt_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant that generates image prompts based on text."},
            {"role": "user", "content": f"Generate a detailed image prompt for AI News Pod cover art based on this transcript. Include a male host and a female host, and the words 'AI News Pod'. Transcript: {json.dumps(transcript)}"}
        ]
    )
    
    prompt = prompt_response.choices[0].message.content
    log("IMAGE_PROMPT", f"Generated image prompt: {prompt}")
    # Generate the image using the prompt
    image_path = generate_default_image(temp_folder, prompt=prompt)
    
    video_path = create_video("combined_dialogue.mp3", image_path, "dialogue_transcript.json")
    log("OUTPUT", f"Final video saved as: {video_path}")
    
    log("PROCESS_END", "Video generation process completed successfully!")

if __name__ == "__main__":
    main()
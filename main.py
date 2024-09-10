import os
import uuid
import json
import requests
import time
from openai import OpenAI
from pydub import AudioSegment

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_dialogue():
    print("Reading content from rawtext.md...")
    with open("rawtext.md", "r") as file:
        content = file.read()

    print("Generating dialogue using OpenAI GPT-4...")
    start_time = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Based on the following content, generate a dialogue discussing the top 5 news of the day. Include an introduction by a host, and then a discussion between two voices: Alex (male) and Sarah (female). The host should only speak at the start, and introduce the headline topics Alex and Sarah will talk about. Alex should introduce the news (mentioning the Discords they are from) and answer questions, while Sarah should make funny/amusing observations and ask follow-up questions for Alex to answer. Give credit to the Discord communities discussing these topics.\n\nContent: {content}"},
            {"role": "user", "content": f"Alex is a 35-year-old former software engineer turned tech journalist. He has a Ph.D. in Computer Science from MIT and spent 5 years working at Google before pursuing his passion for tech reporting. Alex is known for his in-depth knowledge and no-nonsense approach to tech news. He's an avid rock climber and often uses climbing metaphors in his explanations. His catchphrase is 'Let's scale this tech mountain!' and his favorite AI lab is DeepMind.\n\nSarah is a 29-year-old stand-up comedian with a degree in Communications from NYU. She fell into tech journalism by accident when her comedy podcast about ridiculous tech gadgets went viral. Sarah brings a fresh, humorous perspective to tech news, often pointing out the absurd and making witty pop culture references. She's a passionate gamer and often relates tech news to video game scenarios. Her catchphrase is 'Well, that's more confusing than a Rubik's Cube in a blender!' and her favorite AI lab is OpenAI, mainly because she finds their name 'adorably on-the-nose'."}
        ],
        functions=[
            {
                "name": "generate_dialogue",
                "description": "Generate a dialogue discussing the top 5 news of the day",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dialogue": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "speaker": {"type": "string", "enum": ["Host", "Alex", "Sarah"]},
                                    "text": {"type": "string"}
                                },
                                "required": ["speaker", "text"]
                            }
                        }
                    },
                    "required": ["dialogue"]
                }
            }
        ],
        function_call={"name": "generate_dialogue"}
    )
    end_time = time.time()
    print(f"Dialogue generated in {end_time - start_time:.2f} seconds")

    return response.choices[0].message.function_call.arguments

def text_to_speech_file(text: str, voice_id: str, temp_folder: str, history: list) -> tuple:
    print(f"Converting text to speech for voice {voice_id}...")
    start_time = time.time()
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }

    data = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    if history:
        data["history"] = history[-3:]  # Use up to 3 previous generations

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        # Generating a unique file name for the output MP3 file
        save_file_path = os.path.join(temp_folder, f"{uuid.uuid4()}.mp3")

        # Writing the audio to a file
        with open(save_file_path, "wb") as f:
            f.write(response.content)

        end_time = time.time()
        print(f"Audio file saved: {save_file_path} (generated in {end_time - start_time:.2f} seconds)")

        # Get the generation ID from the response headers
        generation_id = response.headers.get("x-request-id")

        # Return the path of the saved audio file and the generation ID
        return save_file_path, generation_id
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None, None

def combine_audio_files(file_paths, output_file):
    print("Combining audio files...")
    start_time = time.time()
    combined = AudioSegment.empty()
    for file_path in file_paths:
        audio = AudioSegment.from_mp3(file_path)
        combined += audio
    combined.export(output_file, format="mp3")
    end_time = time.time()
    print(f"Audio files combined in {end_time - start_time:.2f} seconds")

def main():
    print("Starting the dialogue generation and text-to-speech process...")
    dialogue_json = generate_dialogue()
    dialogue = json.loads(dialogue_json)['dialogue']
    
    # Create a unique temp folder
    temp_folder = f"temp_{uuid.uuid4()}"
    os.makedirs(temp_folder, exist_ok=True)
    print(f"Created temporary folder: {temp_folder}")
    
    voice_host_id = "ThT5KcBeYPX3keUQqHPh"  # Charlie pre-made voice
    voice_alex_id = "pNInz6obpgDQGcFmaJgB"  # Adam pre-made voice for Alex
    voice_sarah_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel pre-made voice for Sarah

    audio_files = []
    history_host = []
    history_alex = []
    history_sarah = []

    print(f"Processing {len(dialogue)} dialogue lines...")
    for i, line in enumerate(dialogue):
        print(f"Processing line {i+1}/{len(dialogue)}: {line['speaker']}")
        if line['speaker'] == "Host":
            voice_id = voice_host_id
            history = history_host
        elif line['speaker'] == "Alex":
            voice_id = voice_alex_id
            history = history_alex
        else:  # Sarah
            voice_id = voice_sarah_id
            history = history_sarah
        
        audio_file, generation_id = text_to_speech_file(line['text'], voice_id, temp_folder, history)
        if audio_file:
            audio_files.append(audio_file)
            history.append({"text": line['text'], "generation_id": generation_id})
        else:
            print(f"Failed to generate audio for line {i+1}")

    # Combine audio files
    if audio_files:
        output_file = "combined_dialogue.mp3"
        combine_audio_files(audio_files, output_file)
        print(f"Combined audio saved as: {output_file}")
    else:
        print("No audio files were generated successfully.")

    # Output the dialogue to a text file
    dialogue_output_file = "dialogue_transcript.txt"
    with open(dialogue_output_file, "w") as f:
        for line in dialogue:
            f.write(f"{line['speaker']}: {line['text']}\n\n")
    print(f"Dialogue transcript saved as: {dialogue_output_file}")

    print("Process completed successfully!")

if __name__ == "__main__":
    main()




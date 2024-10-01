import os
import uuid
import json
import requests
import time
import subprocess
from datetime import datetime
from openai import OpenAI
from pydub import AudioSegment
import io
import concurrent.futures
from pydub.utils import mediainfo  # {{ edit_1 }}

# basemodel = "o1-mini"
basemodel = "gpt-4o"
max_workers = 3  # cartesia default is 3. you have to upgrade to use more.

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

def validate_api_keys():
    required_keys = {
        "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "CARTESIA_API_KEY": CARTESIA_API_KEY
    }

    for key_name, key_value in required_keys.items():
        if not key_value:
            raise ValueError(f"{key_name} is empty or not set. Please ensure all required API keys are properly configured.")

validate_api_keys()

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def log(category, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{category.upper()}] {message}"
    print(log_message)
    with open("ai_news_pod.log", "a") as log_file:
        log_file.write(log_message + "\n")

def generate_dialogue():
    log("INPUT", "Reading content from rawtext.md...")
    with open("rawtext.md", "r") as file:
        content = file.read()

    log("BRAINSTORM", "Generating important news stories and discussion topics...")
    brainstorm_response = openai_client.chat.completions.create(
        model=basemodel,
        messages=[
            {"role": "user", "content": "You are an AI assistant tasked with brainstorming important tech news stories and discussion topics. Focus on new models, data, news, rumors and hot topics. Ignore mundane support or debugging issues."},
            {"role": "user", "content": f"Based on the following content, brainstorm the top 5 most important and interesting tech news stories or discussion items. Highlight important technical details - key dates, numbers, benchmarks, model names, people, companies, funding, rumors. For each topic, provide a brief explanation of why it's significant and how it relates to AI Engineering, machine learning, or tech innovation.\n\nContent: {content}"}
        ]
    )

    brainstormed_topics = brainstorm_response.choices[0].message.content
    log("BRAINSTORM", "Topics generation completed.")
    log("BRAINSTORM_OUTPUT", brainstormed_topics)

    log("QUESTION_GEN", "Generating key questions for each topic...")
    questions_response = openai_client.chat.completions.create(
        model=basemodel,
        messages=[
            {"role": "user", "content": "You are an AI assistant tasked with generating insightful and detailed explanations about tech news items."},
            {"role": "user", "content": f"Based on the following brainstormed topics, generate 2-3 key explanations for each topic that a technical AI Engineer reader or listener already familiar with the AI basics, might want to walk away with. Highlight important technical details - key dates, numbers, benchmarks, model names, people, companies, funding, rumors. These explanations should be thought-provoking, slightly humorous, and encourage detailed technical explanations from Sarah.\n\nBrainstormed Topics:\n{brainstormed_topics}"}
        ]
    )

    brainstormed_questions = questions_response.choices[0].message.content
    log("QUESTION_GEN", "Questions generation completed.")
    log("QUESTION_GEN_OUTPUT", brainstormed_questions)

    log("DIALOGUE_GEN", "Generating dialogue using OpenAI GPT-4...")
    start_time = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with generating a dialogue about tech news."},
            {"role": "user", "content": f"Based on the following brainstormed news items, questions, and original content, generate a dialogue discussing the top 5 news of the day for a show called AI News Pod. Include an introduction by a host (Charlie) mentioning today's month and day, {datetime.now().strftime('%m-%d')}, and then a discussion between two voices: Karan (male) and Sarah (female). The host (Charlie) should only speak briefly at the start, just mentioning the date and major topics (not introducing himself or the Karan or Sarah), and then at each change of topic, and introduce the headline news and facts that Karan and Sarah will then discuss. Sarah should introduce the news (mentioning the sources they are from) and answer questions, while Karan should make funny/amusing but technical observations for an AI Engineer audience and ask follow-up questions for Sarah to answer. Use the brainstormed questions as a guide for Karan's inquiries. Give credit to the source discussing these topics. End with Charlie again telling listeners to send feedback to @smol_ai on Twitter. You have a history of saying the catchphrases too much - say them a MAXIMUM of once in the whole conversation. If in doubt, don't.\n\nBrainstormed Topics:\n{brainstormed_topics}\n\nBrainstormed Questions:\n{brainstormed_questions}\n\nOriginal Content:\n{content}"},
            {"role": "user", "content": f"Sarah is a 35-year-old AI engineer. She has a Ph.D. in Computer Science from MIT and spent 7 years working as a researcher at Google DeepMind. Sarah is known for her in-depth knowledge and no-nonsense approach to tech news. She's an avid rock climber and often uses climbing metaphors in her explanations, but also loves cooking Thai food and surfing. Her catchphrase is 'What a time to be alive!' and her favorite AI lab is DeepMind.\n\nKaran is a 60-year-old Indian comedian with a degree in Communications from NYU. He fell into tech journalism by accident when his comedy podcast about ridiculous tech gadgets went viral. Karan brings a fresh, humorous perspective to tech news, often pointing out the absurd and making witty pop culture references. He's a passionate gamer and often relates tech news to video game scenarios, famous movies and tv shows or science fiction/fantasy books. His catchphrase is 'Super easy, barely an inconvenience!' and his favorite AI lab is OpenAI, mainly because he finds their name 'kind of ironic'."}
        ],
        functions=[
            {
                "name": "generate_dialogue",
                "description": "Generate a dialogue discussing the top 5 news of the day. Focus on technical details, and first lead with all known facts, and then end with some opinions. First the host should briefly introduce the overall day. Then for each news item, the host reads out the factual headline, then Karan and Sarah should discuss the news, trading a few questions back and forth.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dialogue": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "speaker": {"type": "string", "enum": ["Host", "Karan", "Sarah"]},
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
    log("DIALOGUE_GEN", f"Dialogue generated in {end_time - start_time:.2f} seconds")

    return response.choices[0].message.function_call.arguments

def text_to_speech_file(name: str, text: str, voice_id: str, temp_folder: str, history: list, use_cartesia: bool = False, progress: tuple = None) -> tuple:
    if progress:
        current, total = progress
        log("TTS_PROGRESS", f"Converting text to speech for voice {voice_id} ({current}/{total})...")
    else:
        log("TTS", f"Converting text to speech for voice {voice_id}...")
    start_time = time.time()

    if use_cartesia:
        # Use Cartesia API for Karan and Sarah
        url = "https://api.cartesia.ai/tts/bytes"
        headers = {
            "Cartesia-Version": "2024-06-10",
            "X-API-Key": CARTESIA_API_KEY,
            "Content-Type": "application/json"
        }
        data = {
            "transcript": text + "   ",  # cartesia docs say to add a space at the end of the text to prevent cutting off the last word
            "model_id": "sonic-english",
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 44100}
        }

        response = requests.post(url, headers=headers, json=data, stream=True)

        if response.status_code == 200:
            save_file_path = os.path.join(temp_folder, f"{current if progress else ''}-{name}-{uuid.uuid4()}.mp3".strip('-'))

            # Use ffmpeg to convert the raw audio to MP3
            ffmpeg_command = [
                'ffmpeg',
                '-f', 'f32le',
                '-ar', '44100',
                '-ac', '1',
                '-i', 'pipe:0',
                '-acodec', 'libmp3lame',
                '-b:a', '192k',
                save_file_path
            ]

            process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(input=response.content)

            if process.returncode != 0:
                log("TTS_ERROR", f"FFmpeg error: {stderr.decode()}")
                return None, None, 0.0  # {{ edit_2 }}

            generation_id = None  # Cartesia doesn't provide a generation ID

            # Get duration using pydub
            audio = AudioSegment.from_mp3(save_file_path)
            duration_sec = len(audio) / 1000.0  # Duration in seconds
        else:
            log("TTS_ERROR", f"Cartesia API error: {response.status_code} - {response.text}")
            return None, None, 0.0  # {{ edit_3 }}
    else:
        # Use ElevenLabs for Charlie (Host)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        data = {
            "text": text + " ",  # cartesia docs say to add a space at the end of the text to prevent cutting off the last word
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        if history:
            data["history"] = history[-3:]

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            save_file_path = os.path.join(temp_folder, f"{current if progress else ''}-{name}-{uuid.uuid4()}.mp3".strip('-'))
            # save_file_path = os.path.join(temp_folder, f"{name}-{uuid.uuid4()}.mp3")
            with open(save_file_path, "wb") as f:
                f.write(response.content)
            generation_id = response.headers.get("x-request-id")

            # Get duration using pydub
            audio = AudioSegment.from_mp3(save_file_path)
            duration_sec = len(audio) / 1000.0  # Duration in seconds
        else:
            log("TTS_ERROR", f"Error: {response.status_code} - {response.text}")
            return None, None, 0.0  # {{ edit_4 }}

    end_time = time.time()
    log("TTS", f"Audio file saved: {os.path.basename(save_file_path)} (generated in {end_time - start_time:.2f} seconds)")
    return save_file_path, generation_id, duration_sec  # {{ edit_5 }}

def combine_audio_files(file_paths, output_file):
    log("AUDIO_COMBINE", "Combining audio files...")
    start_time = time.time()
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=200)  # 200ms of silence
    for file_path in file_paths:
        audio = AudioSegment.from_mp3(file_path)
        combined += audio + silence  # Add audio clip followed by 300ms silence
    combined = combined[:-200]  # Remove the last silence
    combined.export(output_file, format="mp3")
    end_time = time.time()
    log("AUDIO_COMBINE", f"Audio files combined with 300ms gaps in {end_time - start_time:.2f} seconds")

def main():
    log("PROCESS_START", "Starting the dialogue generation and text-to-speech process...")
    dialogue_json = generate_dialogue()
    dialogue = json.loads(dialogue_json)['dialogue']

    temp_folder = f"temp_{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{uuid.uuid4()}"
    os.makedirs(temp_folder, exist_ok=True)
    log("TEMP_FOLDER", f"Created temporary folder: {temp_folder}")

    voice_host_id = "IKne3meq5aSn9XLyUdCD"  # Charlie pre-made voice (ElevenLabs)
    voice_karan_id = "638efaaa-4d0c-442e-b701-3fae16aad012"  # Replace with actual Cartesia voice ID for Karan
    voice_sarah_id = "79a125e8-cd45-4c13-8a67-188112f4dd22"  # Replace with actual Cartesia voice ID for Sarah

    audio_files = {}  # {{ edit_6 }}
    history_host = []
    history_karan = []
    history_sarah = []

    log("DIALOGUE_PROCESS", f"Processing {len(dialogue)} dialogue lines...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                text_to_speech_file,
                line['speaker'],
                line['text'],
                voice_host_id if line['speaker'] == "Host" else (voice_karan_id if line['speaker'] == "Karan" else voice_sarah_id),
                temp_folder,
                history_host if line['speaker'] == "Host" else (history_karan if line['speaker'] == "Karan" else history_sarah),
                False if line['speaker'] == "Host" else True,
                (i+1, len(dialogue))
            ): i for i, line in enumerate(dialogue)
        }
        # Initialize list to hold transcript with timestamps
        transcript_with_timestamps = []  # {{ edit_7 }}
        current_time = 0.0  # {{ edit_8 }}

        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            line = dialogue[i]
            try:
                audio_file, generation_id, duration = future.result()
                if audio_file:
                    audio_files[i] = audio_file  # {{ edit_9 }}
                    transcript_with_timestamps.append({
                        "speaker": line['speaker'],
                        "text": line['text'],
                        "start_time": current_time,
                        "end_time": current_time + duration,
                        "duration": duration  # {{ edit_10 }} Added duration field
                    })  # ... existing code ...
                    current_time += duration  # {{ edit_11 }}
                    if line['speaker'] == "Host":
                        history_host.append({"text": line['text'], "generation_id": generation_id})
                    elif line['speaker'] == "Karan":
                        history_karan.append({"text": line['text'], "generation_id": generation_id})
                    else:
                        history_sarah.append({"text": line['text'], "generation_id": generation_id})
                else:
                    log("TTS_FAIL", f"Failed to generate audio for line {i+1}/{len(dialogue)}")
            except Exception as exc:
                log("TTS_ERROR", f"Line {i+1}/{len(dialogue)} generated an exception: {exc}")

    if audio_files:
        # Combine audio files in the correct order
        ordered_audio_files = [audio_files[i] for i in range(len(dialogue)) if i in audio_files]
        combine_audio_files(ordered_audio_files, "combined_dialogue.mp3")
        log("OUTPUT", "Combined audio saved as: combined_dialogue.mp3")

        # Save the transcript with timestamps
        with open("dialogue_transcript.json", "w") as f:
            json.dump(transcript_with_timestamps, f, indent=4)
        log("OUTPUT", "Dialogue transcript with timestamps saved as: dialogue_transcript.json")

        log("OUTPUT", "Audio and transcript files generated. Run video.py to create the final video.")
    else:
        log("AUDIO_ERROR", "No audio files were generated successfully.")

    log("PROCESS_END", "Process completed successfully!")

if __name__ == "__main__":
    # Track start time at the beginning of the process
    start_time = time.time()
    main()

    # Calculate and log the total time elapsed
    end_time = time.time()
    total_time = end_time - start_time
    log("TOTAL_TIME", f"Total time elapsed: {total_time:.2f} seconds")
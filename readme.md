# smol notebook lm

Convert text to speech using the `ElevenLabs` client. Ensure `rawtext.md` is updated, all environment keys are loaded, and outputs are available in `.mp3`, `.txt`, and `.log` formats.


## requirements

- make sure ffmpeg is installed
- `brew install imagemagick`

### Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables:**
   ```bash
   export ELEVENLABS_API_KEY=your_api_key
   export OPENAI_API_KEY=your_api_key
   export CARTESIA_API_KEY=your_api_key # note that this thing generates a lot of tokens. we used up 52k cahracters just developing this.
   ```

3. **Update `rawtext.md`:**
   - Add or modify the text you want to convert to speech.

4. **Run the Script:**
   ```bash
   python main.py
   ```

### Optional Video Generation

To generate a video from the audio and transcript, you can use the `video.py` script. This script uses the MoviePy library to combine the audio and image, and the OpenAI library to generate a default image using DALL·E.

#### Requirements

- Make sure you have the `moviepy` and `openai` libraries installed. You can install them using pip:
  ```bash
  pip install moviepy openai
  ```

#### Setup

1. **Set Environment Variables:**
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```

2. **Run the Script:**
   ```bash
   python video.py
   ```

#### Outputs

- **Video:** `final_video.mp4` file

### Notes

- The `video.py` script assumes that the `combined_dialogue.mp3` and `dialogue_transcript.txt` files are present in the same directory.
- The script generates a default image using DALL·E and resizes it to 1080x1080 pixels.
- The script combines the audio and image to create a video, and adds captions using the transcript.
- The final video is saved as `final_video.mp4` in the same directory.



### Outputs

- **Audio:** `.mp3` files
- **Transcript:** `.txt` files
- **Logs:** `.log` files

### Examples

- [Sample MP3](examples/combined_dialogue.mp3)
- [Transcript](examples/dialogue_transcript.txt)
```

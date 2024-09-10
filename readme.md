# main.py

This is the main script for the project. It uses the `ElevenLabs` client to convert text to speech. The text to be converted, the voice settings, and the output format are all configurable.

The script first initializes the `ElevenLabs` client with the API key. It then uses the `text_to_speech.convert` method to convert the text to speech. The voice ID, streaming latency optimization, and output format are all set as parameters. The text to be converted is a quote from the movie "Forrest Gump". The voice settings are also configured to add a touch of style to the speech.

The script is designed to be run from the command line, with the API key provided as an environment variable.

### Installation and Setup

To use this project, follow these steps:

1. Install the required libraries by running `pip install elevenlabs`.
2. Set up your ElevenLabs API key as an environment variable named `ELEVENLABS_API_KEY`.
3. Run the script using `python main.py`.



# KeyPhraseSpotting
Using Nemo ASR to break existing speech data and then filter split segments using Whisper model


arguments
1. inputfolder: folder with wav files in it
2. keywords: list of keywords, 3000 for now (fetched from internet)
3. outputfolder: folder where snippet audio files will be stored
4. manifestfile: file with audio-paths and text

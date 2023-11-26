import nemo.collections.asr as nemo_asr
import whisper

import numpy as np
# Import audio processing library
import librosa
# We'll use this to listen to audio
from IPython.display import Audio, display

from plotly import graph_objects as go

from pydub import AudioSegment
import soundfile as sf

import glob
import string
import json
import os


whisper_model = whisper.load_model("base")
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name='QuartzNet15x5Base-En', strict=False)


def onewordaudio(inputfolder,
                 keywords,
                 outputfolder,
                 manifestfile):

    os.system("mkdir -p " + outputfolder)
    
    audiofiles = glob.glob(inputfolder+"/*.wav")
    keywords = open(keywords, "r").read().split("\n")

    manifest = open(manifestfile, "w")
    all_transcripts = asr_model.transcribe(paths2audio_files=audiofiles)

    words_kept = 0
    for filename in audiofiles:

        files = [filename]
        transcript = asr_model.transcribe(paths2audio_files=files)[0]
        signal, sample_rate = librosa.load(filename, sr=None)

        # softmax implementation in NumPy
        def softmax(logits):
            e = np.exp(logits - np.max(logits))
            return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

        # let's do inference once again but without decoder
        logits = asr_model.transcribe(files, logprobs=True)[0]
        probs = softmax(logits)

        # 20ms is duration of a timestep at output of the model
        time_stride = 0.02

        # get model's alphabet
        labels = list(asr_model.decoder.vocabulary) + ['blank']
        labels[0] = 'space'

        # get timestamps for space symbols
        spaces = []

        state = ''
        idx_state = 0

        if np.argmax(probs[0]) == 0:
            state = 'space'

        for idx in range(1, probs.shape[0]):
            current_char_idx = np.argmax(probs[idx])
            if state == 'space' and current_char_idx != 0 and current_char_idx != 28:
                spaces.append([idx_state, idx-1])
                state = ''
            if state == '':
                if current_char_idx == 0:
                    state = 'space'
                    idx_state = idx

        if state == 'space':
            spaces.append([idx_state, len(pred)-1])
        
        #Then we can split original audio signal into separate words. It is worth to mention that all timestamps have a delay (or an offset) depending on the model. We need to take it into account for alignment.
        # calibration offset for timestamps: 180 ms
        offset = -0.18

        # split the transcript into words
        words = transcript.split()

        # cut words
        pos_prev = 0
        for j, spot in enumerate(spaces):

            if words[j] in keywords:
                pos_end = offset + (spot[0]+spot[1])/2*time_stride

                audio_snippet = signal[int(pos_prev*sample_rate):int(pos_end*sample_rate)]
                audio_snippet_file = outputfolder + "/" + str(words_kept) + ".wav"
                
                sf.write(audio_snippet_file, audio_snippet, 16000)
                whisper_output = whisper_model.transcribe(audio_snippet_file)['text'].lower()
                whisper_output = whisper_output.translate(str.maketrans('', '', string.punctuation)).strip()
                if whisper_output == words[j]:
                    
                    sample = {}
                    sample["audio_filepath"] = audio_snippet_file
                    sample['text'] = words[j]
                    manifest.write(json.dumps(sample) + '\n')
                    words_kept += 1
                else:
                    os.system("rm " + audio_snippet_file)
                
                pos_prev = pos_end
                

    manifest.close()

if __name__ == "__main__":
    onewordaudio(inputfolder="/audio_datasets/EN_libre/LibriSpeech/dev-clean-wav",
                    keywords="keywords.txt",
                    outputfolder="/audio_datasets/split_data",
                    manifestfile="/audio_datasets/split_data.json")

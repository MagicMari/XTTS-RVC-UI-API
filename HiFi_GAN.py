import numpy as np
#mport librosa
import soundfile as sf
#from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.tts.models import Tacotron2Model
import torchaudio
#from torchaudio import transforms
#import torch
#import nemo.collections.asr as nemo_asr

def ini_hifigan():
    global spec_generator
    global model
    #spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
    spec_generator = Tacotron2Model.from_pretrained("tts_en_tacotron2")
    model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")

def tts_to_file(text, sav_path):
    parsed = spec_generator.parse(text)
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    audio = model.convert_spectrogram_to_audio(spec=spectrogram)
    audio_numpy = audio.squeeze().detach().cpu().numpy().astype(np.float32)
    sf.write(sav_path, audio_numpy, 22050, format='WAV')




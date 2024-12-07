import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import time
import HiFi_GAN

def openvoice_setup():
    global device
    global output_dir
    global tone_color_converter
    ckpt_converter = 'checkpoints_v2/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = 'outputs_v2'

    tone_color_converter = ToneColorConverter(f'./OpenVoice/{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'./OpenVoice/{ckpt_converter}/checkpoint.pth')

    os.makedirs(output_dir, exist_ok=True)

def load_se(reference_speaker, language):
    global model
    model = TTS(language="EN_NEWEST", device=device)
    global target_se
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)
    #model = TTS(language=language, device=device) Moved to setup
    speaker_ids = model.hps.data.spk2id
    global speaker_id
    global speaker_key
    if language == "EN_NEWEST":
        speaker_id = speaker_ids["EN-Newest"]
        speaker_key = "en-newest"
    else:
        speaker_id = speaker_ids["EN-US"]
        speaker_key = "en-us"
    global source_se
    source_se = torch.load(f'./OpenVoice/checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)


#text = "Did you ever hear a folk tale about a giant turtle?"
#language = "EN_NEWEST"
def tts_to_file(text, reference_speaker, language, file_path, hifi_gan):
    #reference_speaker = '/home/mari/Documents/TTS/XTTS-RVC-UI-API/voices/example_reference.wav' # This is the voice you want to clone
    speed = 1.0
    src_path = f'{output_dir}/tmp.wav'
    startMelo = time.time()
    if(hifi_gan == True):
        HiFi_GAN.tts_to_file(text, src_path)
    else:
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
    endMelo = time.time()
    print("Melo:")
    print(endMelo - startMelo)

    startConv = time.time()
    save_path = f'{file_path}'
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)
    endConv = time.time()
    print("Convert:")
    print(endConv - startConv)


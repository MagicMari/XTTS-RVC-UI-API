import torch
#from TTS.api import TTS
from rvc import Config, load_hubert, get_vc, rvc_infer
import gc , os, requests
from pathlib import Path
import json
import time
import HiFi_GAN

voices = []
rvcs = []
langs = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]


def download_models():
	rvc_files = ['hubert_base.pt', 'rmvpe.pt']

	for file in rvc_files: 
		if(not os.path.isfile(f'./models/{file}')):
			print(f'Downloading{file}')
			r = requests.get(f'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/{file}')
			with open(f'./models/{file}', 'wb') as f:
					f.write(r.content)

	#xtts_files = ['vocab.json', 'config.json', 'dvae.path', 'mel_stats.pth', 'model.pth']

	#for file in xtts_files:
	#	if(not os.path.isfile(f'./models/xtts/{file}')):
	#		print(f'Downloading {file}')
	#		r = requests.get(f'https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/{file}')
	#		with open(f'./models/xtts/{file}', 'wb') as f:
	#			f.write(r.content)
				


def startup():
	global device
	global config
	global hubert_model
	#global tts
	[Path(_dir).mkdir(parents=True, exist_ok=True) for _dir in ['./models/xtts', './voices', './rvcs']]
	download_models()
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	print("Device: " + device) 

	config = Config(device, device != 'cpu')
	hubert_model = load_hubert(device, config.is_half, "./models/hubert_base.pt")
	#tts = TTS(model_path="./models/xtts", config_path='./models/xtts/config.json').to(device)

	import melo_tts
	global tts_melo
	tts_melo = melo_tts
	tts_melo.melo_init(1.0, "auto", "EN")
 
	import openvoice_tts
	global tts_openvoice
	tts_openvoice = openvoice_tts
	tts_openvoice.openvoice_setup()

	HiFi_GAN.ini_hifigan()

	#openvoice_tts.load_se(language="EN_NEWEST", reference_speaker="./voices/Female-01.wav")


def get_rvc_voices():
	global voices 
	voices = os.listdir("./voices")
	global rvcs
	rvcs = list(filter(lambda x:x, os.listdir("./rvcs")))
	return [rvcs, voices]

#deprecated
#def runtts(rvc, voice, text, pitch_change, index_rate, language): 
#   audio = tts.tts_to_file(text=text, speaker_wav="./voices/" + voice, language=language, file_path="./output.wav")
#    voice_change(rvc, pitch_change, index_rate)
#    return ["./output.wav" , "./outputrvc.wav"]
#deprecated
#def runtts(rvc, text, pitch_change, index_rate, language):
#	modelname = str(rvc).replace(".pth","")
#	audio = tts.tts_to_file(text=text, speaker_wav="./rvcs/"+ modelname + "/" + modelname + ".wav", language=language, file_path="./output.wav")
#	voice_change(rvc, pitch_change, index_rate)
#	return ["./output.wav" , "./outputrvc.wav"]

def load_profile(profile_name):
	global isLoaded
	try:
		jsonFile = open("./voice_models.json")
		voices = json.load(jsonFile)
		ref_voice = "./voices/" + voices[profile_name]["reference_voice"]
		lang = voices[profile_name]["language"]
		tts_openvoice.load_se(reference_speaker=ref_voice, language=lang)
		jsonFile.close()
		isLoaded = True
	except:
		isLoaded = False

def runtts(voice_model ,text, pitch_change, index_rate):
	if(isLoaded == False):
		return
	voice_model = json.loads(voice_model)
	voice_model = dict(voice_model)
	modelname = voice_model["rvc_path"]
	rvc = modelname + ".pth"
	reference_voice = voice_model["reference_voice"]
	tts_model = voice_model["model"]
	lang = voice_model["language"]
	startTTS = time.time()
	audio = tts_openvoice.tts_to_file(text=text, reference_speaker="./voices/"+ reference_voice, language=lang, file_path="./output.wav", hifi_gan=False)
	endTTS = time.time()
	print("TTS:")
	print(endTTS - startTTS)
	startRVC = time.time()
	voice_change(rvc, pitch_change, index_rate)
	endRVC = time.time()
	print("RVC:")
	print(endRVC - startRVC)
	#Apply HiFi-GAN then output
	return ["./output.wav" , "./outputrvc.wav"]

def main():
	startup()
	get_rvc_voices()
	print(rvcs)


# delete later

class RVC_Data:
	def __init__(self):
		self.current_model = {}
		self.cpt = {}
		self.version = {}
		self.net_g = {} 
		self.tgt_sr = {}
		self.vc = {} 

	def load_cpt(self, modelname, rvc_model_path):
		if self.current_model != modelname:
				print("Loading new model")
				del self.cpt, self.version, self.net_g, self.tgt_sr, self.vc
				self.cpt, self.version, self.net_g, self.tgt_sr, self.vc = get_vc(device, config.is_half, config, rvc_model_path)
				self.current_model = modelname

rvc_data = RVC_Data()

def voice_change(rvc, pitch_change, index_rate):
	modelname = os.path.splitext(rvc)[0]
	print("Using RVC model: "+ modelname)
	rvc_model_path = "./rvcs/" + modelname + "/" + rvc
	rvc_index_path = "./rvcs/" + modelname + "/" + modelname + ".index" if os.path.isfile("./rvcs/" + modelname + "/" + modelname +".index") and index_rate != 0 else ""

	if rvc_index_path != "" :
		print("Index file found!")

	#load_cpt(modelname, rvc_model_path)
	#cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)
	rvc_data.load_cpt(modelname, rvc_model_path)
	
	rvc_infer(
		index_path=rvc_index_path, 
		index_rate=index_rate, 
		input_path="./output.wav", 
		output_path="./outputrvc.wav", 
		pitch_change=pitch_change, 
		f0_method="rmvpe", 
		cpt=rvc_data.cpt, 
		version=rvc_data.version, 
		net_g=rvc_data.net_g, 
		filter_radius=3, 
		tgt_sr=rvc_data.tgt_sr, 
		rms_mix_rate=0.25, 
		protect=0, 
		crepe_hop_length=0, 
		vc=rvc_data.vc, 
		hubert_model=hubert_model
	)
	gc.collect()

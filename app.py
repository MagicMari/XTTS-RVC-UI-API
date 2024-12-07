import gradio as gr
import core_tts
import json
import os
import time

# File to store the voice configurations
config_file = "voice_models.json"
core_tts.main()

# A dictionary to store voice configurations in memory
voice_models = {}

def load_configurations():
    global voice_models
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            voice_models = json.load(file)
            print(f"Loaded configurations from {config_file}")
    else:
        # Create an empty JSON file if it doesn't exist
        with open(config_file, 'w') as file:
            json.dump({}, file, indent=4)
        print(f"Created a new configuration file: {config_file}")
        voice_models = {}

def save_configurations():
    with open(config_file, 'w') as file:
        json.dump(voice_models, file, indent=4)
        print(f"Saved configurations to {config_file}")

def main():
    core_tts.get_rvc_voices()
    load_configurations()

    def refresh_dropdowns():
        rvcs, voices = core_tts.get_rvc_voices()
        updated_choices = {
            "model": ["XTTS", "MeloTTS", "OpenVoice"],
            "rvc": ["None"] + [str(rvc) for rvc in rvcs],
            "ref_voice": ["None"] + [str(ref_voice) for ref_voice in voices],
            "lang": ["None", "EN_NEWEST", "EN"],
            "existing_voices": ["None"] + list(map(str, voice_models.keys()))
        }

        return [
            gr.update(choices=["None"] + list(map(str, voice_models.keys())), value="None"),  # Existing Voices dropdown in Voice Configurator
            gr.update(choices=updated_choices["model"], value="XTTS"),  # Model dropdown in Voice Configurator
            gr.update(choices=updated_choices["rvc"], value="None"),  # RVC dropdown in Voice Configurator
            gr.update(choices=updated_choices["ref_voice"], value="None"),  # Reference Voice dropdown in Voice Configurator
            gr.update(choices=updated_choices["lang"], value="None"),  # Language dropdown in Voice Configurator
            gr.update(choices=["None"] + list(map(str, voice_models.keys())), value="None"),  # Voice Model dropdown in TTS
        ]

    def save_voice_configuration(name, model, rvc, ref_voice, lang):
        if name:
            voice_models[name] = {
                "model": str(model),
                "rvc_path": str(rvc),
                "reference_voice": str(ref_voice),
                "language": str(lang)
            }
            save_configurations()  # Save to file whenever a new config is added
            print(f"Saved voice configuration for '{name}': {voice_models[name]}")
            return "Voice configuration saved!"
        else:
            return "Please enter a name for the voice configuration."

    def load_voice_configuration(name, model_dropdown, rvc_dropdown, ref_voice_dropdown, lang_dropdown):
        if name in voice_models:
            config = voice_models[name]
            model, rvc, ref_voice, lang = str(config["model"]), str(config["rvc_path"]), str(config["reference_voice"]), str(config["language"])

            # Update dropdowns dynamically if the values do not exist
            def update_dropdown_if_needed(dropdown, value):
                choices = dropdown.choices
                if value not in choices:
                    choices.append(value)
                return gr.update(choices=choices, value=value)

            return [
                update_dropdown_if_needed(model_dropdown, model),
                update_dropdown_if_needed(rvc_dropdown, rvc),
                update_dropdown_if_needed(ref_voice_dropdown, ref_voice),
                update_dropdown_if_needed(lang_dropdown, lang)
            ]
        else:
            return ["None", "None", "None", "None"]

    def submit_tts(model_name, text_input, pitch, index_rate):
        start = time.time()
        if model_name in voice_models:
            config = voice_models[model_name]
            json_config = json.dumps(config)
            print(f"Submitting TTS with configuration: {json_config}")
            # Pass JSON config to core_tts
            audio, rvc_audio = core_tts.runtts(json_config, text_input, pitch, index_rate)
            end = time.time()
            print("Total time:")
            print(end - start)
            return audio, rvc_audio
        else:
            return None, None

    def load_selected_voice(model_name):
        if model_name and model_name != "None":
            core_tts.load_profile(model_name)
            return f"Loaded voice model: {model_name}"
        else:
            return "Please select a valid voice model to load."

    with gr.Blocks(title='TTS RVC UI') as interface:
        with gr.Tab("TTS"):
            with gr.Row():
                gr.Markdown("""
                    # XTTS RVC UI
                """)
            with gr.Row():
                with gr.Column():
                    model_dropdown_tts = gr.Dropdown(choices=["None"] + list(map(str, voice_models.keys())), label='Voice Model')
                    text_input = gr.Textbox(placeholder="Write here...")
                    submit_button = gr.Button(value='Submit')
                    load_model_button = gr.Button(value="Load Voice Model")  # New button to load the voice model
                    with gr.Row():
                        pitch_slider = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="Pitch")
                        index_rate_slider = gr.Slider(minimum=0, maximum=1, value=0.75, step=0.05, label="Index Rate")
                with gr.Column():
                    audio_output = gr.Audio(label="TTS result", type="filepath", interactive=False)
                    rvc_audio_output = gr.Audio(label="RVC result", type="filepath", interactive=False)

            submit_button.click(
                inputs=[model_dropdown_tts, text_input, pitch_slider, index_rate_slider],
                outputs=[audio_output, rvc_audio_output],
                fn=submit_tts
            )

            load_model_button.click(  # Configure the new button
                inputs=[model_dropdown_tts],
                outputs=[gr.Textbox(label="Load Status")],  # Display the status of the loading process
                fn=load_selected_voice
            )

        with gr.Tab("Voice Configurator"):
            rvcs, voices = core_tts.get_rvc_voices()
            model_name_input = gr.Textbox(label="Voice Model Name")
            model_dropdown_vc = gr.Dropdown(choices=["XTTS", "MeloTTS", "OpenVoice"], label='Model')
            rvc_dropdown_vc = gr.Dropdown(choices=[str(rvc) for rvc in rvcs], label='RVC')
            ref_voice_dropdown = gr.Dropdown(choices=[str(ref_voice) for ref_voice in voices], label='Reference Voice')
            lang_dropdown_vc = gr.Dropdown(choices=["EN_NEWEST", "EN"], label='Language')
            existing_voices_dropdown = gr.Dropdown(choices=["None"] + list(map(str, voice_models.keys())), label='Existing Voices')
            save_button = gr.Button(value='Save Voice')

            # When selecting an existing voice model, load its configuration
            existing_voices_dropdown.change(
                fn=lambda name: load_voice_configuration(name, model_dropdown_vc, rvc_dropdown_vc, ref_voice_dropdown, lang_dropdown_vc),
                inputs=[existing_voices_dropdown],
                outputs=[model_dropdown_vc, rvc_dropdown_vc, ref_voice_dropdown, lang_dropdown_vc]
            )

            save_button.click(
                fn=save_voice_configuration, 
                inputs=[model_name_input, model_dropdown_vc, rvc_dropdown_vc, ref_voice_dropdown, lang_dropdown_vc], 
                outputs=[gr.Textbox(value="")]  # You can use a textbox to display the save confirmation message
            )

        # Refresh button to update all dropdowns
        refresh_button = gr.Button(value='Refresh')
        refresh_button.click(fn=refresh_dropdowns, outputs=[
            model_dropdown_tts,
            existing_voices_dropdown,
            model_dropdown_vc,
            rvc_dropdown_vc,
            ref_voice_dropdown,
            lang_dropdown_vc
        ])

    interface.launch(server_name="0.0.0.0", server_port=8889, quiet=True)

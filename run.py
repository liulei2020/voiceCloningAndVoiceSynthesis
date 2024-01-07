from pathlib import Path
import soundfile as sf
import os
from paddlespeech.t2s.exps.syn_utils import get_am_output
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_predictor
from paddlespeech.t2s.exps.syn_utils import get_voc_output
import numpy as np

def generate_audio(am_inference_dir, voc_inference_dir, wav_output_dir, device, text_dict):
    
    print(f"am_inference_dir: {am_inference_dir}")
    print(f"voc_inference_dir: {voc_inference_dir}")
    print(f"wav_output_dir: {wav_output_dir}")
    print(f"device: {device}")
    print(f"text_dict: {text_dict}")
    
    # frontend
    frontend = get_frontend(
        lang="mix",
        phones_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
        tones_dict=None
    )

    # am_predictor
    am_predictor = get_predictor(
        model_dir=am_inference_dir,
        model_file="fastspeech2_mix" + ".pdmodel",
        params_file="fastspeech2_mix" + ".pdiparams",
        device=device
    )

    # voc_predictor
    voc_predictor = get_predictor(
        model_dir=voc_inference_dir,
        model_file="pwgan_aishell3" + ".pdmodel",
        params_file="pwgan_aishell3" + ".pdiparams",
        device=device
    )

    output_dir = Path(wav_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = list(text_dict.items())
    print(f"sentences: {sentences}")

    utt_id, sentence = sentences[0]
    wav_paths = []  # List to store the paths of the generated audio files

    if len(sentence) > 30:
        # Split the sentence into chunks of size 30
        chunks = [sentence[i:i+30] for i in range(0, len(sentence), 30)]
        
        # Create a new dictionary with numbered keys
        text_dict_chunked = {i: chunk for i, chunk in enumerate(chunks)}
        
        for idx, chunk in text_dict_chunked.items():
            am_output_data = get_am_output(
                input=chunk,
                am_predictor=am_predictor,
                am="fastspeech2_mix",
                frontend=frontend,
                lang="mix",
                merge_sentences=True,
                speaker_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
                spk_id=0
            )
            wav = get_voc_output(voc_predictor=voc_predictor, input=am_output_data)
            # Save the file
            wav_path = output_dir / (f"{utt_id}_{idx}.wav")
            sf.write(wav_path, wav, samplerate=24000)
            wav_paths.append(wav_path)

        # Merge the generated audio files into a single WAV file
        merged_wav = sf.read(wav_paths[0], dtype='int16')[0]  # Read the first file
        for path in wav_paths[1:]:
            merged_wav = np.concatenate((merged_wav, sf.read(path, dtype='int16')[0]))  # Concatenate the remaining files

        # Save the merged file
        merged_wav_path = output_dir / f"{utt_id}_merged.wav"
        sf.write(merged_wav_path, merged_wav, samplerate=24000)
        
        # Remove individual audio files after merging
        for path in wav_paths:
            os.remove(path)

        # Return the paths of the generated audio files
        return merged_wav_path

    else:
        # Process the original sentence
        am_output_data = get_am_output(
            input=sentence,
            am_predictor=am_predictor,
            am="fastspeech2_mix",
            frontend=frontend,
            lang="mix",
            merge_sentences=True,
            speaker_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
            spk_id=0
        )
        wav = get_voc_output(voc_predictor=voc_predictor, input=am_output_data)
        # Save the file
        wav_path = output_dir / (utt_id + ".wav")
        sf.write(wav_path, wav, samplerate=24000)

        # Return the path of the generated audio file
        return wav_path

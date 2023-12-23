from pathlib import Path
import soundfile as sf
import os
from paddlespeech.t2s.exps.syn_utils import get_am_output
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_predictor
from paddlespeech.t2s.exps.syn_utils import get_voc_output

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

    merge_sentences = True
    fs = 24000
    for utt_id, sentence in sentences:
        am_output_data = get_am_output(
            input=sentence,
            am_predictor=am_predictor,
            am="fastspeech2_mix",
            frontend=frontend,
            lang="mix",
            merge_sentences=merge_sentences,
            speaker_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
            spk_id=0
        )
        wav = get_voc_output(voc_predictor=voc_predictor, input=am_output_data)
        # Save the file
        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=fs)


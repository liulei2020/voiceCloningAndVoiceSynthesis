from pathlib import Path
import soundfile as sf
import os
from paddlespeech.t2s.exps.syn_utils import get_am_output
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_predictor
from paddlespeech.t2s.exps.syn_utils import get_voc_output

# 在其他环境中，记得修改下面这两个变量的路径
am_inference_dir = "D:/Codeeeee/PYTHON/voiceclone/interface/LO"
voc_inference_dir = "D:/app/pwgan_aishell3_static_1.1.0"  # 这里以 pwgan_aishell3 为例子

# 音频生成的路径，修改成你音频想要保存的路径
wav_output_dir = "D:/Codeeeee/PYTHON/voiceclone/results"

# 选择设备[gpu / cpu]，这里以GPU为例子， 
device = "cpu"

# 想要生成的文本和对应文件名

text_dict = {
    "1": "今天天气真不错，欢迎和我一起玩。",
    "2": "我认为跑步给我的身体带来了健康。",
}


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
    device=device)

# voc_predictor
voc_predictor = get_predictor(
    model_dir=voc_inference_dir,
    model_file="pwgan_aishell3" + ".pdmodel",    # 这里以 pwgan_aishell3 为例子，其它模型记得修改此处模型名称
    params_file="pwgan_aishell3" + ".pdiparams",
    device=device)

output_dir = Path(wav_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

sentences = list(text_dict.items())

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
        spk_id=0, )
    wav = get_voc_output(
            voc_predictor=voc_predictor, input=am_output_data)
    # 保存文件
    sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=fs)
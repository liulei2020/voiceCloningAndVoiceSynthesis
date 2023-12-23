import gradio as gr
import os
from run import generate_audio
import datetime

# 设置项目所在的基本路径
base_dir = "D:\\deeplearning\\voiceclone\\"  # 替换为实际的项目基本路径

# 获取interface目录下的文件夹列表
interface_folders = [os.path.join(base_dir, "interface", folder) for folder in os.listdir(os.path.join(base_dir, "interface")) if os.path.isdir(os.path.join(base_dir, "interface", folder))]

# 获取encoder目录下的文件夹列表
encoder_folders = [os.path.join(base_dir, "encoder", folder) for folder in os.listdir(os.path.join(base_dir, "encoder")) if os.path.isdir(os.path.join(base_dir, "encoder", folder))]



# 设置默认的result_file_name为当前时间戳的yyyyMMddHHmmSS格式
default_result_file_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


# 定义一个包装函数，用于传递json_data参数
def generate_audio_wrapper(model, encoder, result_file_name, inference_device, text_content):
    result_dir = os.path.join(base_dir, "results")
    # 处理用户输入的text_content和result_file_name，组装成字典
    text_dict = {result_file_name: text_content}
    # 调用实际的generate_audio函数
    return generate_audio(model, encoder,result_dir, inference_device, text_dict)

# 创建Gradio界面
iface = gr.Interface(
    fn=generate_audio_wrapper,  # Use a wrapper function
    inputs=[
        gr.Dropdown(interface_folders, label="选择模型"),
        gr.Dropdown(encoder_folders, label="选择编码器"),
        gr.Textbox(default_result_file_name, label="输入合成语音的文件名"),  # 使用默认值
        gr.Radio(["CPU", "GPU"], label="选择推理设备"),
        gr.Textbox("", lines=5, label="输入待合成的文字"),  # 空的文本框，用于用户输入
    ],
    outputs=[
        gr.Audio(type="numpy", label="合成语音")
    ],
)


# 启动Gradio界面
iface.launch()



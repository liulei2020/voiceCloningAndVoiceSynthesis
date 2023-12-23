from flask import Flask, render_template, request
from flask_cors import CORS
from run import generate_audio

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        am_inference_dir = request.form['am_inference_dir']
        voc_inference_dir = request.form['voc_inference_dir']
        wav_output_dir = request.form['wav_output_dir']
        device = request.form['device']
        text_dict_str = request.form['text_dict']

        # Convert text_dict_str to a dictionary
        text_dict = eval(text_dict_str)

        # Call the generate_audio function
        generate_audio(am_inference_dir, voc_inference_dir, wav_output_dir, device, text_dict)

        return render_template('index.html', success=True)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()

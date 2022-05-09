import os
import threading

from flask import Flask, json, render_template, request, redirect, url_for
from pyngrok import ngrok
from dotenv import load_dotenv

from predict import get_prediction
from helpers import format_class_name

load_dotenv()

os.environ["FLASK_ENV"] = "development"

app = Flask(__name__)
PORT = os.getenv("FLASK_PORT")


ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(PORT).public_url
print(f' * ngrok tunnel "{public_url}" -> "http://127.0.0.1:{PORT}"')

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return redirect(url_for('index'))

        img_bytes = uploaded_file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        return render_template('prediction.html', class_id=class_id,
                               class_name=class_name)
    return render_template('index.html')


@app.route('/payload', methods=['GET', 'POST'])
def gh_webhook():
    if request.method == 'POST':
        if request.headers['Content-Types'] == 'application/json':
            return json.dumps(request.json)
    return "GitHub webhook endpoint"


if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()
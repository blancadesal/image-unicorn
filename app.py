import os

from flask import Flask, render_template, request, redirect, url_for

from predict import get_prediction
from helpers import format_class_name

app = Flask(__name__)


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


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
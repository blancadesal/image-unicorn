import logging
from flask import render_template, request, redirect, url_for

from helpers import format_class_name
from predict import get_prediction
from project import create_app

logging.basicConfig(format='%(name)s - %(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = create_app()


@app.route("/")
def hello():
    return "Use the /predict endpoint for image classification"


@app.route('/predict', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return redirect(url_for('index'))

        img_bytes = uploaded_file.read()
        logger.debug("Image bytes are in!")
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        logger.debug("Prediction is done!")
        class_name = format_class_name(class_name)
        return render_template('prediction.html', class_id=class_id,
                               class_name=class_name)
    return render_template('index.html')
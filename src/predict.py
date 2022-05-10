
import logging
import json

from helpers import get_model, transform_image

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

model = get_model()
if model:
    logger.debug("Got the model!")

imagenet_class_index = json.load(open('imagenet_class_index.json'))

def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        logger.debug("We have a tensor!")
        outputs = model.forward(tensor)
    except Exception:
        return 0, 'error'
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
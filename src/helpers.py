import io
import logging

from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

logging.basicConfig(format='%(name)s - %(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_model():
    model = models.densenet121(pretrained=True)
    model.eval()
    return model

def transform_image(image_bytes):
    logger.debug('Trying to transform this here image!')
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    if image:
        logger.debug("Got image inside transform!")
    result_image = my_transforms(image).unsqueeze(0)
    logger.debug("result_image OK!")
    return result_image


def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name
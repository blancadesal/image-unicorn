import io
import logging
from PIL import Image
from pathlib import Path

from torchvision import models
import torchvision.transforms as transforms

logging.basicConfig(format='%(name)s - %(asctime)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

image_path = Path('/data/project/image-unicorn/repo/image-unicorn/test-images/dog.jpg')
image = Image.open(image_path)

def get_model():
    model = models.densenet121(pretrained=True)
    model.eval()
    return model

model = get_model()

def transform_image(image_bytes):
    my_transforms1 = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        ])
    my_transforms2 = transforms.Compose([transforms.ToTensor(),
                                        ])
    my_transforms3 = transforms.Compose([transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    logger.debug("Image is ready for transform!")
    result_image1 = my_transforms1(image)
    logger.debug("Transform1 applied! (Resize, CenterCrop)")
    result_image2 = my_transforms2(result_image1)
    logger.debug("Transform2 applied! (ToTensor)")
    result_image3 = my_transforms3(result_image2)
    logger.debug("Transform3 applied! (Normalize)")
    return result_image3.unsqueeze(0)



tensor = transform_image(image)

outputs = model.forward(tensor)
_, y_hat = outputs.max(1)
predicted_idx = str(y_hat.item())

print(predicted_idx)
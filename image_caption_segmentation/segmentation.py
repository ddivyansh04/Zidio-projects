import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image

model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

def segment_image(image: Image.Image) -> Image.Image:
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    colormap = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
    segmented_image = colormap[output_predictions]
    return Image.fromarray(segmented_image)

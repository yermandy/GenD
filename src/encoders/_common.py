import requests
import torch
from PIL import Image


def inference(model):
    for name, param in model.named_parameters():
        print(name, param.shape)
    print()

    print(model)
    print()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    preprocessed = [model.preprocess(image) for image in [image, image]]
    preprocessed = torch.stack(preprocessed)
    outputs = model(preprocessed)

    print(outputs.shape)

from torchvision import transforms
from PIL import Image
import torch


def resize(img_path, style='s1'):
    if style == 's2':
        width = 128
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif style == 's1':
        width = 256
        mean = torch.tensor([0.0, 0.0, 0.0])
        std = torch.tensor([1.0, 1.0, 1.0])

    style_transform = transforms.Compose([transforms.Resize((width, width)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])

    img = Image.open(img_path)
    final_image = style_transform(img).reshape(1, 3, width, width)
    return final_image


def denormalisation(x, style='s1'):
    if style == 's2':
        width = 128
        mean = 0.5
        std = 0.5
    elif style == 's1':
        width = 256
        mean = 0.0
        std = 1.0
    x = x.reshape(3, width, width)
    return x * std + mean


# import argparse
# import time
# from tkinter.tix import IMAGE
# from torch.autograd import Variable
# from torchvision.transforms import ToTensor, ToPILImage

import torch
from PIL import Image

from torchvision import transforms
from torchvision import models

from data_utils import get_std, get_mean


if __name__ == '__main__':

    MODEL_NAME = 'best_model.pth'
    IMAGE_NAME = 'image.jpg'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_NAME))

    image = Image.open(IMAGE_NAME)

    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(get_mean(image), get_std(image))])

    image = transforms(image)

    # print(image)

    image = image.to(device)

    input = image.unsqueeze(0)

    out = model(input)

    print(torch.max(out.data), 1)

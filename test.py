import torch

from torchvision import transforms
from torchvision import models

from PIL import Image
import argparse

from data_utils import mean, std

parser = argparse.ArgumentParser(
    description='Test MOIVS (Car image classification))')
parser.add_argument(
    '--model_name', default='./epochs/best_model.pth', type=str, help='path of .pth file')
parser.add_argument('--image_name', default='./data/img.jpg',
                    type=str, help='image path')
parser.add_argument('--label', default='./car_list',
                    type=str, help='label file path')

opt = parser.parse_args()

MODEL_NAME = opt.model_name
IMAGE_NAME = opt.image_name
CAR_LIST = opt.label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

car_list = torch.load(CAR_LIST)
num_classes = len(car_list)
model = models.resnet50(
    pretrained=False, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_NAME))

img = Image.open(IMAGE_NAME)
image = []
image.append((img, ' '))

transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])

img = transform(img)

img = img.to(device)

input = img.unsqueeze(0)

out = model(input)

_, result = torch.max(out.data, 1)
print(car_list[result[0]])

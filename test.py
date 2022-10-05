import torch
from PIL import Image

from torchvision import transforms
from torchvision import models

from data_utils import get_std, get_mean


if __name__ == '__main__':

    MODEL_NAME = './epochs/best_model.pth'
    IMAGE_NAME = './data/img.jpg'
    CAR_LIST = './car_list'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_NAME))
    car_list = torch.load(CAR_LIST)

    img = Image.open(IMAGE_NAME)
    image = []
    image.append((img, ' '))

    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(get_mean(image), get_std(image))])

    img = transform(img)

    img = img.to(device)

    input = img.unsqueeze(0)

    out = model(input)

    _, result = torch.max(out.data, 1)
    print(result[0])
    print(car_list[result[0]])

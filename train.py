import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from torchvision import transforms

from tqdm import tqdm

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, get_mean, get_std


def train(model, params):
    loss_function = params["loss_function"]
    train_dataloader = params["train_dataloader"]
    val_dataloader = params["val_dataloader"]
    device = params["device"]

    best_accurancy = 0

    for epoch in range(0, num_epochs):
        for i, data in enumerate(tqdm(train_dataloader, desc='train'), 0):
            # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 이전 batch에서 계산된 가중치를 초기화
            optimizer.zero_grad()

            # forward + back propagation 연산
            outputs = model(inputs)
            train_loss = loss_function(outputs, labels)
            train_loss.backward()
            optimizer.step()

        # val accuracy 계산
        total = 0
        correct = 0
        accuracy = []
        for i, data in enumerate(tqdm(val_dataloader, desc='val'), 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 결과값 연산
            outputs = model(inputs)
            # print(torch.max(outputs.data, 1))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss = loss_function(outputs, labels).item()
            accuracy.append(100 * correct/total)

        # 학습 결과 출력
        print('Epoch: %d/%d, Train loss: %.6f, Val loss: %.6f, Accuracy: %.2f' %
              (epoch+1, num_epochs, train_loss.item(), val_loss, 100*correct/total))

        # 모델 파라미터 저장
        if best_accurancy < 100*correct/total:
            best_accurancy = 100*correct/total
            torch.save(model.state_dict(),
                       './best_model_epoch_%d.pth' % (epoch))


if __name__ == '__main__':
    # 학습 환경 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('********** Device = '+str(device)+' **********')
    # resnet50 모델을 사용
    model = models.resnet50(pretrained=False).to(device)

    # dataset
    print('Data load...')
    train_path = './data/car_sample'
    val_path = './data/car_sample'

    train_dataset = TrainDatasetFromFolder(path=train_path)
    val_dataset = ValDatasetFromFolder(path=val_path)
    print('Data load complete!')

    # 이미지 크기를 임의로 128로 고정한 후, 정규화하는 과정만 진행
    print('Data transform...')
    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(get_mean(train_dataset), get_std(train_dataset))])
    val_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(get_mean(val_dataset), get_std(val_dataset))])

    # trainsform 정의
    train_dataset.transform = train_transforms
    val_dataset.transform = val_transforms
    print('Data transform complete!')

    # dataloader 정의
    print('Define data loader...')
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    print('Define data loader complete!')

    print('Set parameters...')
    lr = 0.0001
    num_epochs = 5
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss().to(device)

    params = {
        'num_epochs': num_epochs,
        'optimizer': optimizer,
        'loss_function': loss_function,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'device': device
    }
    print('Set parameters complete!')

    print('********** Training Start! **********')
    train(model, params)

# from genericpath import isfile
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
        for i, data in enumerate(tqdm(train_dataloader, desc='train\t'), 0):
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
        for i, data in enumerate(tqdm(val_dataloader, desc='val\t'), 0):
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
        print('\nEpoch: %d/%d, Train loss: %.6f, Val loss: %.6f, Accuracy: %.2f\n' %
              (epoch+1, num_epochs, train_loss.item(), val_loss, 100*correct/total))
        # torch.save(model.state_dict(), './epochs/epoch_%d.pth' % (epoch))
        # 모델 파라미터 저장
        if best_accurancy < 100*correct/total:
            best_accurancy = 100*correct/total
            torch.save(model.state_dict(),
                       './epochs/best_model.pth')


if __name__ == '__main__':

    batch = 64
    lr = 0.0001
    num_epochs = 5

    # 학습 환경 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n\n********** Device = '+str(device)+' **********\n\n')
    # resnet50 모델을 사용
    model = models.resnet50(pretrained=False).to(device)

    # dataset
    print('Data load...')
    dataset_path = './data/car_data/'

    train_dataset = TrainDatasetFromFolder(path=dataset_path)
    val_dataset = ValDatasetFromFolder(path=dataset_path)
    print('Data load complete!')
    # print(train_dataset.car_label_num)

    # 이미지 크기를 임의로 128로 고정한 후, 정규화하는 과정만 진행
    print('Data transform...')
    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(get_mean(train_dataset, 'train'), get_std(train_dataset, 'train'))])
    val_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(get_mean(val_dataset, 'val'), get_std(val_dataset, 'val'))])

    # trainsform 정의
    train_dataset.transform = train_transforms
    val_dataset.transform = val_transforms
    print('Data transform complete!')

    # dataloader 정의
    print('Define data loader...')
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    print('Define data loader complete!')

    print('Set parameters...')
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

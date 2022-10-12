import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from torchvision import models
from torchvision import transforms

from tqdm import tqdm
import pandas as pd
import argparse

from data_utils import dataset_load, std, mean

parser = argparse.ArgumentParser(
    description='Train MOVIS (Car image classification)')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int,
                    help='train epoch number')
parser.add_argument('--path', default='./data/car_data',
                    type=str, help='path of dataset')


def train(model, params):
    loss_function = params["loss_function"]
    train_dataloader = params["train_dataloader"]
    val_dataloader = params["val_dataloader"]
    device = params["device"]

    # statistics를 위한 딕셔너리 정의 및 경로설정
    out_path = './statistics/'
    results = {'Train loss': [], 'Val loss': [],
               'Val Accuracy': [], 'Train Accuracy': []}
    best_results = {'Epoch': [], 'Train loss': [],
                    'Val loss': [], 'Val Accuracy': [], 'Train Accuracy': []}

    best_accurancy = 0

    for epoch in range(0, NUM_EPOCHS):
        train_total = 0
        train_correct = 0
        train_accuracy = 0
        for data in tqdm(train_dataloader, desc='train'):
            # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 이전 batch에서 계산된 가중치를 초기화
            optimizer.zero_grad()

            # forward + back propagation 연산
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_accuracy = (100 * train_correct/train_total)
            train_loss = loss_function(outputs, labels)
            train_loss.backward()
            optimizer.step()

        # val accuracy 계산
        total = 0
        correct = 0
        accuracy = 0
        for data in tqdm(val_dataloader, desc='val  '):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 결과값 연산
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss = loss_function(outputs, labels).item()
            accuracy = (100 * correct/total)

        scheduler.step()

        # 학습 결과 출력
        print('Epoch: %d/%d, Train loss: %.6f, Val loss: %.6f, Train Accuracy: %.2f, Val Accuracy: %.2f\n' %
              (epoch+1, NUM_EPOCHS, train_loss.item(),
               val_loss, train_accuracy, accuracy))

        # 모델 파라미터 저장
        results['Train loss'].append(train_loss.item())
        results['Val loss'].append(val_loss)
        results['Val Accuracy'].append(accuracy)
        results['Train Accuracy'].append(train_accuracy)

        # 10 epochs마다 statistics 저장
        if epoch % 10 == 0 and epoch != 0:
            data_frame = pd.DataFrame(
                data={'Train loss': results['Train loss'], 'Val loss': results['Val loss'], 'Train Accuracy': results['Train Accuracy'], 'Val Accuracy': results['Val Accuracy']}, index=range(1, epoch+2))
            data_frame.to_csv(out_path+'train_results.csv',
                              index_label='Epoch')

        # 정확도가 가장 높은 모델의 parameters와 statistics 저장
        if best_accurancy < 100*correct/total:
            best_results['Epoch'].append(epoch+1)
            best_results['Train loss'].append(train_loss.item())
            best_results['Val loss'].append(val_loss)
            best_results['Val Accuracy'].append(accuracy)
            best_results['Train Accuracy'].append(train_accuracy)

            best_accurancy = 100*correct/total
            torch.save(model.state_dict(),
                       './epochs/best_model.pth')
            data_frame = pd.DataFrame(
                data={'Epoch': best_results['Epoch'], 'Train loss': best_results['Train loss'],
                      'Val loss': best_results['Val loss'], 'Train Accuracy': best_results['Train Accuracy'], 'Val Accuracy': best_results['Val Accuracy']},
                index=range(1, len(best_results['Train loss'])+1))
            data_frame.to_csv(
                out_path+'best_train_results.csv', index_label='Num')


if __name__ == '__main__':
    opt = parser.parse_args()

    # 학습 환경 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n\n********** Device = '+str(device)+' **********\n\n')

    BATCH_SIZE = opt.batch
    LEARNING_RATE = opt.lr
    NUM_EPOCHS = opt.epochs
    DATASET_PATH = opt.path

    # dataset
    print('Setting data...')

    # 이미지 크기를 임의로 128로 고정한 후, 정규화하는 과정만 진행
    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean,
                                                                std=std)])
    val_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean,
                                                              std=std)])

    train_dataset = dataset_load(
        path=DATASET_PATH, type='train', transform=train_transforms)
    val_dataset = dataset_load(
        path=DATASET_PATH, type='val', transform=val_transforms)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Set ResNet50...')
    num_classes = len(torch.load('./car_list'))

    model = models.resnet50(
        pretrained=False, num_classes=num_classes).to(device)

    print('Set train parameters...')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss().to(device)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=0.001)

    params = {
        'NUM_EPOCHS': NUM_EPOCHS,
        'optimizer': optimizer,
        'loss_function': loss_function,
        'scheduler': scheduler,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'device': device
    }

    print('********** Training Start! **********')

    train(model, params)

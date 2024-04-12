import torch
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import json
from utils import *

from datasets import DrinksDataset
from models import DrinksClassificationModel

#Считывание конфига в словарь
json_file_path = 'config.json'
with open(json_file_path, 'r') as j:
    config = json.loads(j.read())
config = Dotdict(config)            #Класс в utils для удобного использования словаря


#Трансформер для изображений
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([config.img_size, config.img_size]),
])

train_df = DrinksDataset('data/train.csv', config, transformer)
test_df = DrinksDataset('data/test.csv', config, transformer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#Создание модели
epochs = 50
model = DrinksClassificationModel(config)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
model.to(device)
print(model)

train_dataloader = DataLoader(train_df, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_df, batch_size=config.batch_size, shuffle=True)

check_points_path = 'check_points/'

#Тренировка
for ep in tqdm(range(epochs)):
    model.train()
    all_loss = 0    #Переменная для подсчета суммарного лосса по батчам
    for img, labels in tqdm(train_dataloader):
        img = img.to(device)
        labels =  labels.to(device)
        labels = nn.functional.one_hot(labels, config.count_class).to(torch.float)

        optimizer.zero_grad()
        outputs = model(img)

        loss = criterion(outputs, labels)
        all_loss += loss
        loss.backward()
        optimizer.step()

        #_, predicted = torch.max(outputs.data, 1)

    print(f'\nEpoch: {ep}')
    print(f'Loss: {all_loss / len(train_dataloader)}')
    torch.save(model.state_dict(), check_points_path + f"ep_{ep}")

    model.eval()
    with torch.no_grad():
        all_test_label = torch.tensor([]).to(torch.int).to(device)     #Тензоры реальных меток и предиктов
        all_test_predict = torch.tensor([]).to(torch.int).to(device)   # для вычисления метрик
        for test_img, test_label in test_dataloader:
            test_img = test_img.to(device)
            test_label = test_label.to(device)
            outputs = model(test_img)
            _, predicted = torch.max(outputs.data, 1)

            all_test_label = torch.cat((all_test_label, test_label))
            all_test_predict = torch.cat((all_test_predict, predicted))

        print('F1: ' + str(f1_score(all_test_label.to('cpu'), all_test_predict.to('cpu'), average=None)))
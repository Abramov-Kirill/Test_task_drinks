import torch
import pandas as pd
from PIL import Image


class DrinksDataset(torch.utils.data.Dataset):
  def __init__(self, csv_path, config, transform):
    self.csv_path = csv_path
    self.transform = transform
    self.df = pd.read_csv(csv_path)
    self.config = config

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):

    image = Image.open(self.df.iloc[index]['Path'])
    image = self.transform(image)

    label = torch.tensor(self.df.iloc[index]['Label'])

    return image, label

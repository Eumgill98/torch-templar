import cv2
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, info={'path':'path', 'label':'label'} ,transform=None, train_mode=True):
        self.df = df
        self.image = df[info['path']].values
        if train_mode:
            self.label = df[info['label']].values
        self.transform = transform
        self.train_mode = train_mode
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.image[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.train_mode:
            label = self.label[idx]
            return image, label
        
        return image
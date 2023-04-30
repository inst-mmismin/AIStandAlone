import os 
from PIL import Image 
from torch.utils.data import Dataset

class Dog_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        self.dog_class = os.listdir(self.data_path)
        self.dog_class_idx = {dog_class:idx for idx, dog_class in enumerate(self.dog_class)}

        self.image_path = []
        for dog_class in self.dog_class:
            image_path = [ os.path.join(self.data_path, dog_class, image_name) 
                          for image_name in os.listdir(os.path.join(self.data_path, dog_class))]
            self.image_path += image_path
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        dog_class = os.path.basename(os.path.dirname(image_path))
        dog_class_idx = self.dog_class_idx[dog_class]
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
        
        return image, dog_class_idx
from PIL import Image
import os
from datasets.datasets_utils import getItem
from torch.utils.data import Dataset

# https://discuss.pytorch.org/t/how-to-load-images-without-using-imagefolder/59999

class COWS(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
        self.image_filenames = [f for f in self.image_filenames if f.endswith('.jpg')]
        self.image_filenames.sort()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # FYI: Need PIL image ordered as (batch, channel, h, w)
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        img = Image.open(img_path).convert("RGB")

        return getItem(img, target=None, transform=self.transform, training_mode='SSL')

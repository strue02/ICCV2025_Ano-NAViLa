import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd

class word_Dataset(Dataset):
    """
    Word Dataset for text pool.

    Args:
        csv_path (str): Path to the CSV file containing words.
        header (str): Column name to use ('Normal' or 'Abnormal').

    Returns:
        Each item is a single word from the specified column.
    """
    def __init__(self, csv_path, header='Normal'):
        self.samples = []
        word_csv = pd.read_csv(csv_path)
        for index, row in word_csv.iterrows():
            self.samples.append(row[header])

    def __getitem__(self, index):
        word = self.samples[index]
        return word

    def __len__(self):
        return len(self.samples)

class normal_img_Dataset(Dataset):
    """
    Image Dataset for training or validation (normal images only).

    Args:
        top_path (str): Top folder path containing normal images.
        transform (callable, optional): Optional transform to be applied on a sample.
                                        Defaults to ToTensor() if None.

    Returns:
        tuple: (img, img_name)
            img      : Transformed image tensor.
            img_name : Original image filename.
    """
    def __init__(self, top_path, transform=None):
        self.samples = []
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        for root, dirs, files in os.walk(top_path):
            for file in files:
                if file.lower().endswith(".png"):
                    file_path = os.path.join(root, file)
                    self.samples.append((file, file_path))

        print("[Normal images dataset] Total: {}".format(len(self.samples)), end="\t")
        print()

    def __getitem__(self, index):
        img_name, img_path = self.samples[index]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.samples)

class WSI_patch_Dataset(Dataset):
    """
    Dataset for all image patches of a single WSI (Whole Slide Image).

    Args:
        folder_path (str): Path to a folder named after a specific WSI (e.g., "WSI_001").
                           This folder should contain all patches of that WSI.
        transform (callable, optional): Transformations to apply to each patch image.
                                        Defaults to ToTensor() if not provided.

    Notes on patch file naming:
        - Each patch file name should ideally follow the format:
          {WSI_name}_{w_coord}_{h_coord}_{abnormal_percent}.png
            - {WSI_name}: Name of the WSI
            - {w_coord}, {h_coord}: Top-left pixel coordinates of the patch in the WSI
            - {abnormal_percent}: Percent of abnormal region in the patch (optional)
        - If your patch naming scheme is different, adjust this section accordingly
          to match your file naming convention.
    """
    def __init__(self, folder_path, transform=None):
        self.samples = []
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        for file in os.listdir(folder_path):
            if file.endswith('.png'):
                file_path = os.path.join(folder_path, file)
                self.samples.append((file, file_path))

        print("[WSI patch dataset] Total: {}".format(len(self.samples)), end="\t")
        print()

    def __getitem__(self, index):
        img_name, img_path = self.samples[index]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.samples)
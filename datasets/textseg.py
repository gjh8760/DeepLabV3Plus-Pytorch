import os
import glob
import sys
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        
        cmap[i] = np.array([r, g, b])
    
    cmap = cmap/255 if normalized else cmap
    return cmap


class TextSeg(data.Dataset):
    
    cmap = voc_cmap()

    def __init__(self, root, image_set='train', transform=None):
        
        self.root = root
        self.transform = transform
        self.image_set = image_set
        assert self.image_set in ['train', 'val']

        base_dir = os.path.join(self.root, 'TextSeg', self.image_set)
        self.images = sorted(glob.glob(os.path.join(base_dir, 'image', '*.png')))
        self.masks = sorted(glob.glob(os.path.join(base_dir, 'mask', '*.png')))
        assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        # pixel value 100 -> 1
        target = np.array(target)
        assert target.max() <= 100, "Mask has invalid pixel value. 0: background, 100: text."
        target = (target / 100.).astype(np.uint8)
        target = Image.fromarray(target)
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


# if __name__ == '__main__':
#     dataset = TextSeg(root='./datasets', image_set='train', transform=None)
#     print(len(dataset))
#     for img, mask in dataset:
#         print('')

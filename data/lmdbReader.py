import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
from config import config

class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, alphabet=None):

        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.alphabet = alphabet

    def __len__(self):
        return self.nSamples
    
    # 修改 lmdbReader.py 文件中的 __getitem__ 方法
    def __getitem__(self, index):
        max_retries = 5  # 最大重试次数，避免无限递归
        retry_count = 0
        
        while retry_count < max_retries:
            # 确保索引在有效范围内
            if index >= len(self):
                index = len(self) - 1
            assert index < len(self), f'Error index: {index}, dataset length: {len(self)}'
            
            current_index = index + 1  # 如果 jpg_to_lmdb.py 中索引从 1 开始，这里保持一致
            with self.env.begin(write=False) as txn:
                img_key = f'image_{current_index:09d}'.encode()  # 确保键名格式一致
                imgbuf = txn.get(img_key)

                if imgbuf is None:
                    print(f'Image data not found for key {img_key.decode()}, retrying...')
                    index = (index + 1) % len(self)  # 尝试下一个索引
                    retry_count += 1
                    continue

                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img = Image.open(buf)
                except IOError:
                    print(f'Corrupted image for index {current_index}, retrying...')
                    index = (index + 1) % len(self)  # 尝试下一个索引
                    retry_count += 1
                    continue

                label_key = f'label_{current_index:09d}'.encode()  # 确保键名格式一致
                label_bytes = txn.get(label_key)
                if label_bytes is None:
                    print(f'Label data not found for key {label_key.decode()}, retrying...')
                    index = (index + 1) % len(self)  # 尝试下一个索引
                    retry_count += 1
                    continue
                    
                label = label_bytes.decode('utf-8')

                if label not in self.alphabet:
                    print(f'Label "{label}" not in alphabet, retrying...')
                    index = random.randint(0, len(self) - 1)  # 尝试随机索引
                    retry_count += 1
                    continue

                if len(label) <= 0:
                    print(f'Empty label for index {current_index}, retrying...')
                    index = (index + 1) % len(self)  # 尝试下一个索引
                    retry_count += 1
                    continue

                label += '$'
                label = label.lower()

                if self.transform is not None:
                    img = self.transform(img)

                return (img, label)
        
        # 如果达到最大重试次数，返回一个默认值或抛出异常
        print(f"Failed to retrieve valid data after {max_retries} retries.")
        if hasattr(self, 'default_value'):
            return self.default_value
        else:
            # 创建一个空白图像和默认标签
            img = Image.new('RGB', (32, 32), color='white')
            if self.transform is not None:
                img = self.transform(img)
            return (img, '<pad>$')  # 使用一个特殊的默认标签

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):

        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):

        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


from ctypes import util
from cv2 import IMREAD_GRAYSCALE
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
from torchvision import transforms
from PIL import Image
import cv2
import os
import random
import numpy as np
import yaml
import lmdb
import io
import time
from vidaug import augmentors as va

import copy

import pytorch_lightning as pl
from transformers import XGLMTokenizer
from dataset.utils import Brightness, Color, load_dataset_file, read_lmdb_folder, data_augmentation



class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        if isinstance(Image, PIL.Image.Image):
            Image = np.asarray(Image, dtype=np.uint8)
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class SomeOf(object):
    """
    Selects one augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, clip):
        select = random.choice([0, 1, 2])
        if select == 0:
            return clip
        elif select == 1:
            if random.random() > 0.5:
                return self.transforms1(clip)
            else:
                return self.transforms2(clip)
        else:
            clip = self.transforms1(clip)
            clip = self.transforms2(clip)
            return clip

class S2T_Dataset(Dataset):

    def __init__(self, path, tokenizer, config, phase, max_words=128, training_refurbish=False, resize=256, input_size=224):
        self.config = config
        self.max_words = max_words
        self.training_refurbish = training_refurbish

        self.resize = resize
        self.input_size = input_size
        
        self.raw_data = load_dataset_file(path)

        self.tokenizer = tokenizer
        # print(tokenizer.bos_token_id == 0)
        # print(tokenizer.eos_token_id == 2)
        # print(tokenizer.pad_token_id == 1)

        self.lmdb_path = config['data']['lmdb_path']
        self.phase = phase
        self.max_length = config['data']['max_length']
        
        self.list = [key for key,value in self.raw_data.items()]   

        sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        self.seq = va.Sequential([
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.2)),
            sometimes(va.RandomTranslate(x=10, y=10)),

        ])
        self.seq_color = va.Sequential([
            sometimes(Brightness(min=0.1, max=1.5)),
            sometimes(Color(min=0.1, max=1.5)),
        ])
    def __len__(self):
        # return len(self.raw_data)
        return 10
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        file_name = sample['name']
        # file_name = sample['imgs_path']
        text = sample['text']
        length = sample['length']

        encoding = self.tokenizer(text, truncation=False, add_special_tokens=False ,return_tensors='pt')
        # print(text)
        # print(encoding)
        
        img_sample = self.load_imgs(file_name)
        # print(img_sample.shape)
        return {
            'file_name': file_name,
            'video': img_sample,
            'input_ids': encoding['input_ids'].squeeze(),
        }
        # return file_name, img_sample, encoding['input_ids'].squeeze()
    
    def load_imgs(self, file_name):
        phase, file_name = file_name.split('/')
        folder = os.path.join(self.lmdb_path, phase)
        # print(folder, file_name)
        images = read_lmdb_folder(folder, file_name)
        # print(len(images))
        # print(type(images[0]))
        len_imgs = len(images)
        
        data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])
        
        if len_imgs > self.max_length:
            images = images[:self.max_length]
            len_imgs = len(images)
    
        imgs = torch.zeros(len_imgs,3, self.input_size,self.input_size)
        crop_rect, resize = data_augmentation(resize=(self.resize, self.resize), crop_size=self.input_size, is_train=(self.phase=='train'))
        
        batch_image = []
        for i,img in enumerate(images):
            # print(img.shape)
            img = np.transpose(img, (1, 2, 0))
            # img = np.transpose(img, (0, 1, 2))
            img = Image.fromarray(img)
            batch_image.append(img)
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        # print(file_name)
        # print(len_imgs)
        # print(type(images[0]))
        # exit()

        if self.phase == 'train':
            batch_image = self.seq(batch_image)

        for i, img in enumerate(batch_image):
            img = img.resize(resize)
            img = data_transform(img).unsqueeze(0)
            imgs[i,:,:,:] = img[:,:,crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]]
        
        # print(imgs.shape)
        # print(imgs[0])
        # exit()
        return imgs

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'


def collate_fn(input_batch):
    # Add <bos> token to the beginning of each sequence
    batch = [torch.cat([torch.tensor([0]), seq['input_ids']]) for seq in input_batch]
    
    # Pad the sequences to the maximum length in the batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=1)
    # Create attention masks
    attention_mask = (padded_batch != 1).long()
    
    # Prepare inputs and labels for next token prediction
    inputs = padded_batch[:, :-1]
    labels = padded_batch[:, 1:]
    
    list_of_frames = [seq['video'] for seq in input_batch]
    
    return {
        'input_ids': inputs,
        'attention_mask': attention_mask[:, :-1],
        'labels': labels,
        'list_of_frames': list_of_frames,
    }


class DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            root_text_path,
            qa_csv_path,
            tokenizer_path,
            data_config: dict|str,
            resize=256,
            input_size=224,
            batch_size=1, 
            num_workers=10,
            data_ver=0):
        super().__init__()
        self.text_train = root_text_path + '.train'
        self.text_val = root_text_path + '.dev'
        self.text_test = root_text_path + '.test'

        self.qa_csv_path = qa_csv_path
        self.tokenizer_path = tokenizer_path

        if type(data_config) == str:
            with open(data_config, 'r') as file:
                self.data_config = yaml.safe_load(file)
        else:
            self.data_config = data_config
        
        if data_ver != 0:
            self.data_config['data']['lmdb_path'] = self.data_config['data']['lmdb_path'] + f'_{data_ver}'

        self.resize = resize
        self.input_size = input_size

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        ####################Intialize Tokenizer####################
        self.tokenizer = XGLMTokenizer.from_pretrained(tokenizer_path)
        # Ensure the tokenizer has the necessary special tokens
        # special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str|None = None):
        
        if stage == 'fit' or stage is None:
            # tran and valdiation dataset
            self.train_dataset = S2T_Dataset(path=self.text_train, tokenizer=self.tokenizer, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='train')
            # train_sqa_dataset = SQA_Dataset(path=self.qa_csv_path, tokenizer_path=self.tokenizer_path, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='train')
            # self.train_dataset = CombinedDataset(train_slt_dataset, train_sqa_dataset)

            self.val_dataset = S2T_Dataset(path=self.text_val, tokenizer=self.tokenizer, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='dev')

        if stage == 'test' or stage is None:
            # test dataset
            self.test_dataset = S2T_Dataset(path=self.text_test, tokenizer=self.tokenizer, config=self.data_config, resize=self.resize, input_size=self.input_size, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self):        
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_fn)


if __name__ == "__main__":
    import yaml
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    
    # config = {
    #     'data': {
    #         'lmdb_path': '/home/sobhan/Documents/Code/LLaMA-Adapter/SQA-Lightning/src/sqa/data/lmdb',
    #         'max_length': 300,
    #     }
    # }
    # print(config)
    tokenizer_path = "/home/sobhan/Documents/Code/xglm-1.7B"
    # qa_csv_path = 'src/sqa/data/clean-qa.csv'
    qa_csv_path = None
    root_text_path = '/home/sobhan/Documents/Code/LLaMA-Adapter/SQA-Lightning/src/sqa/data/labels'
    phase = 'train'
    data_module = DataModule(
        root_text_path,
        qa_csv_path,
        tokenizer_path,
        data_config=config,
        batch_size=4,
        num_workers=10,
    )

    data_module.setup()
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    test_dataset = data_module.test_dataset

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    train_dataloader = data_module.train_dataloader()
    # print(dataloader)

    # Example training loop
    for idx, batch in enumerate(train_dataloader):
        print(batch['input_ids'].shape)
        print(batch['input_ids'][0])
        print(batch['labels'].shape)
        print(batch['labels'][0])
        print(batch['attention_mask'].shape)
        print(batch['attention_mask'][0])
        for video in batch['list_of_frames']:
            print(video.shape)
        print('Successfully loaded batch {}'.format(idx))
        break


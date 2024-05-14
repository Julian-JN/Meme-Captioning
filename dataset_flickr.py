import os
import torch
import requests
import json
import pickle
import re
import unicodedata
import itertools

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class FlickrDataset(Dataset):
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3


    def __init__(self, voc_init=True):
        """
        Initialize the MemeDatasetFromFile.
        Args:
            json_file (str): Path to the JSON file containing meme data.
            transform (callable, optional): Optional data transformations (e.g., resizing, normalization).
        """
        # self.mean, self.std = self.calculate_mean_std()
        # self.transform = transforms.Compose([transforms.Resize((300, 300)),  # Example: Resize to 224x224
        #                                      transforms.ToTensor(),
        #                                      transforms.Normalize([0.5471, 0.5182, 0.49696], [0.2940, 0.2934, 0.2992])])
        self.transform = transforms.Compose([transforms.Resize((512, 512)),  # Example: Resize to 224x224
                                             transforms.ToTensor()])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        caption_file = 'flickr' + '/captions.txt'
        self.data = pd.read_csv(caption_file)
        self.imgs = self.data["image"]
        self.captions = self.data["caption"]
        print("There are {} image to captions".format(len(self.data)))
        print(self.data.head())

        self.max_seq_len = 80

        self.word2index = {}
        self.word2count = {}
        self.index2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3:'UNK'}
        self.n_words = 4  # Count SOS and EOS and PAD and UNK

        if voc_init:
            self.voc = self.create_vocab()
            print(len(self.voc))
            print(self.voc)
            f = open("vocab.txt", "w")
            f.write(str(self.voc))
            f.close()
            with open("vocabulary.pkl", "wb") as file:
                pickle.dump(self.voc, file)
        else:
            self.load_vocab()

    def calculate_mean_std(self):
        means_r = []
        means_g = []
        means_b = []
        stds_r = []
        stds_g = []
        stds_b = []

        for filename in os.listdir("flickr/Images/"):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # check for image files
                img = Image.open(os.path.join("flickr/Images/", filename)).convert('RGB')
                img_array = np.array(img) / 255.0  # normalize pixel values to [0, 1]
                means_r.append(np.mean(img_array[:, :, 0]))
                means_g.append(np.mean(img_array[:, :, 1]))
                means_b.append(np.mean(img_array[:, :, 2]))
                stds_r.append(np.std(img_array[:, :, 0]))
                stds_g.append(np.std(img_array[:, :, 1]))
                stds_b.append(np.std(img_array[:, :, 2]))

        overall_mean_r = np.mean(means_r)
        overall_mean_g = np.mean(means_g)
        overall_mean_b = np.mean(means_b)
        overall_std_r = np.mean(stds_r)
        overall_std_g = np.mean(stds_g)
        overall_std_b = np.mean(stds_b)

        overall_mean = [overall_mean_r, overall_mean_g, overall_mean_b]
        overall_std = [overall_std_r, overall_std_g, overall_std_b]
        print(overall_mean)
        print(overall_std)

        return overall_mean, overall_std

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
            if self.word2count[word] >= 2 and word not in self.word2index:  # threshold
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

    def print_info(self):
        print(self.data.head())

    def create_vocab(self):
        meme_captions = self.data["caption"].tolist()
        all_words = meme_captions

        # Normalize each word, and add to vocab
        vocab = self.word2index
        for sentence in all_words:
            normalized_sentence = self.normalize_string(sentence)
            for word in normalized_sentence.split(' '):
                self.addWord(word)
        return vocab

    def load_vocab(self):
        path_vocab = os.path.join(os.getcwd(), 'vocabulary.pkl')
        if os.path.exists(path_vocab):
            # Load the object back from the file
            with open(path_vocab, "rb") as file:
                self.voc = pickle.load(file)
        return

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        # s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
        s = re.sub(r"[^a-zA-Z'!?]+", r" ", s)  # Include apostrophe in the character set
        return s.strip()

    def tokenize_sentence(self, sentence):
        tokenized_sentence = []
        max_length = []

        for text in sentence:
            normalized_text = self.normalize_string(text)
            tokenized_text = [self.voc[word] if word in self.voc else 3 for word in normalized_text.split(' ')]
            tokenized_text.append(self.EOS_token)
            max_length.append(len(tokenized_text))
            max_size = np.zeros(80, dtype=np.int32)
            max_size[:len(tokenized_text)] = tokenized_text
            tokenized_sentence.append(max_size)

        # If multiple captions, choose randomly from choice
        return tokenized_sentence, max(max_length)

    def load_data_from_json(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        matching_captions = self.data.loc[self.data["image"] == img_name, "caption"]
        all_captions = matching_captions.tolist()
        all_captions, max_all_caption = self.tokenize_sentence(all_captions)
        img_location = os.path.join('flickr/Images', img_name)
        image= Image.open(img_location).convert("RGB")
        img_captions, max_caption = self.tokenize_sentence([caption])

        if self.transform:
            image = self.transform(image).to(device)

        return {
            "image_name": img_name,
            "image": image,
            "meme_captions": torch.tensor(np.array(img_captions), dtype=torch.long, device=device),
            "max_caption":max_caption,
            "all_captions": torch.tensor(np.array(all_captions), dtype=torch.long, device=device) # Proper Bleu eval only works in Batch size 1
        }


# Usage example:
if __name__ == "__main__":

    train_dataset = FlickrDataset()  # Add your image transformations if needed
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    print(len(train_dataset))

    instances = 0
    for data in train_dataloader:
        meme_captions = data["all_captions"].squeeze(1)
        print(meme_captions.shape)
        instances += 1
    print(instances)

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
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

class MemeDatasetFromFile(Dataset):
    SOS_token = 0
    EOS_token = 1

    def __init__(self, json_file, voc_init = True):
        """
        Initialize the MemeDatasetFromFile.
        Args:
            json_file (str): Path to the JSON file containing meme data.
            transform (callable, optional): Optional data transformations (e.g., resizing, normalization).
        """
        self.json_file = json_file
        self.json_voc = "data/memes-voc.json"
        self.transform = transforms.Compose([transforms.Resize((256, 256)),  # Example: Resize to 224x224
                                    transforms.ToTensor()])
        self.data = self.load_data_from_json(self.json_file)
        self.data_voc = self.load_data_from_json(self.json_voc)
        self.meme_name = [item['img_fname'] for item in self.data]
        self.titles_voc = [item['title'] for item in self.data_voc]
        # print(self.meme_name)
        # print(len(self.meme_name))
        # print(len(self.titles_voc))

        self.max_seq_len = 80

        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

        if voc_init:
            self.voc = self.create_vocab()
            # print(len(self.voc))
            f = open("vocab.txt", "w")
            f.write(str(self.voc))
            f.close()
            with open("vocabulary.pkl", "wb") as file:
                pickle.dump(self.voc, file)
        else:
            self.load_vocab()

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def create_vocab(self):
        titles = [item['title'] for item in self.data_voc]
        meme_captions = [item['meme_captions'] for item in self.data_voc]
        meme_captions = [x for xs in meme_captions for x in xs]
        img_captions = [item['img_captions'] for item in self.data_voc]
        img_captions = [x for xs in img_captions for x in xs]
        # print(titles)
        all_words = list(itertools.chain(titles, meme_captions, img_captions))
        # print(all_words)

        # Normalize each word, and add to vocab
        vocab = self.word2index
        for sentence in all_words:
            normalized_sentence = self.normalize_string(sentence)
            # print(normalized_sentence)
            for word in normalized_sentence.split(' '):
                self.addWord(word)
        # print(vocab)
        return vocab


    def load_vocab(self):
        path_vocab = os.path.join(os.getcwd(), 'vocabulary.pkl')
        if os.path.exists(path_vocab):
            # Load the object back from the file
            with open(path_vocab, "rb") as file:
                self.voc = pickle.load(file)
        return

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
        return s.strip()

    def tokenize_sentence(self, sentence):
        # print(sentence)
        # print(len(sentence))
        tokenized_sentence = []
        for text in sentence:
            # print(text)
            normalized_text = self.normalize_string(text)
            tokenized_text = [self.voc[word] for word in normalized_text.split(' ')]
            tokenized_text.append(self.EOS_token)
            max_size = np.zeros(self.max_seq_len, dtype=np.int32)
            max_size[:len(tokenized_text)] = tokenized_text
            tokenized_sentence.append(max_size)
        return tokenized_sentence

    def load_data_from_json(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meme_info = self.data[idx]

        # Load image from URL
        # img_url = meme_info["url"]
        # response = requests.get(img_url, stream=True, timeout=5)
        # try:
        #     image = Image.open(BytesIO(response.content)).convert('RGB')
        #     image_path = "img"
        #     name = meme_info["title"]
        #     print(name)
        #     image.save(f"{image_path}/{name}.png")
        #     # plt.imshow(image)
        #     # plt.axis("off")  # Hide axes
        #     # plt.show()
        # except Exception as e:
        #     # print(str(e))
        #     image = 0
        #     title = 0
        #     meme_captions = 0
        #     img_captions = 0
        #     return {
        #         "image": image,
        #         "title": title,
        #         "meme_captions": meme_captions,
        #         "img_captions": img_captions,
        #         "valid": False
        #     }

        image_path = "img_meme/memes/"
        image = Image.open(os.path.join(image_path, self.meme_name[idx])).convert('RGB')
        # plt.imshow(image)
        # plt.axis("off")  # Hide axes
        # plt.show()
        title = self.tokenize_sentence([meme_info["title"]])
        meme_captions = self.tokenize_sentence(meme_info["meme_captions"])
        img_captions = self.tokenize_sentence(meme_info["img_captions"])

        if self.transform:
            image = self.transform(image)
        image = torch.as_tensor(image, dtype=torch.float32, device=device)  # Convert to float32

        # title_ids = np.zeros(self.max_seq_len, dtype=np.int32)
        # meme_captions_ids = np.zeros(self.max_seq_len, dtype=np.int32)
        # img_captions_ids = np.zeros(self.max_seq_len, dtype=np.int32)
        #
        # title_ids[:len(title)] = title
        # meme_captions_ids[:len(meme_captions)] = meme_captions
        # img_captions_ids[:len(img_captions)] = img_captions


        return {
            "image": image,
            "title": torch.tensor(np.array(title), dtype=torch.long, device=device),
            "meme_captions": torch.tensor(np.array(meme_captions), dtype=torch.long, device=device),
            "img_captions": torch.tensor(np.array(img_captions), dtype=torch.long, device=device),
            "valid": True
        }

# Usage example:
if __name__ == "__main__":
    path_test = "data/memes-test.json"
    path_train = "data/memes-trainval.json"

    train_dataset = MemeDatasetFromFile(path_train)  # Add your image transformations if needed
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    instances = 0
    for batch in train_dataloader:
        images = batch["image"]
        if not torch.is_tensor(images) or batch["valid"] == False:
            continue
        else:
            instances += 1
        titles = batch["title"]
        meme_captions = batch["meme_captions"]
        img_captions = batch["img_captions"]
    print(instances)


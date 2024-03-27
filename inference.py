import time
import torch
import wandb
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import MemeDatasetFromFile
from logger import Logger
from models_captioning import EncoderCNN, DecoderLSTM
from utils_functions import save_checkpoint, load_checkpoint

from torch.utils.data import Dataset, DataLoader

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def evaluate(encoder, decoder, input_tensor, caption, voc):
    EOS_token = 1
    with torch.no_grad():
        encoder_outputs = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, caption, None)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(voc.index2word[idx.item()])
    return decoded_words


def test_epoch(dataloader, encoder, decoder, criterion, output_lang):
    total_loss = 0
    batch_loss = 0.0
    total_samples = 0
    for data in dataloader:
        images = data["image"]
        # print(images.shape)
        if not torch.is_tensor(images) or data["valid"] == False:
            continue
        titles = data["title"]
        meme_captions = data["meme_captions"]

        total_samples += 1
        print(total_samples)

        caption = meme_captions[:, 0]
        items = [item.item() for item in caption[0]]
        EOS_token = 1
        decoded_words = []
        for idx in items:
            if idx == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx])
        print("Target")
        print(decoded_words)

        img_captions = data["img_captions"]
        with torch.no_grad():
            encoder_outputs = encoder(images)
            decoder_outputs, _, _ = decoder(encoder_outputs, meme_captions, meme_captions)


        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            meme_captions[:, 0].view(-1)
        )

        text = evaluate(encoder, decoder, images, meme_captions, output_lang)
        print("Predictions")
        print(text)

        total_loss += loss.item()
        batch_loss += loss.item()

    return total_loss / len(dataloader)


def test(test_dataloader, encoder, decoder, logger, output_lang):
    criterion = nn.NLLLoss()

    loss = test_epoch(test_dataloader, encoder, decoder, criterion,
                      output_lang)
    print(loss)


def main():
    # dataset = TranslationDataset('eng', 'fra', True)
    hidden_size = 128
    batch_size = 1
    n_epochs = 20

    path_test = "data/memes-test.json"
    path_train = "data/memes-test.json"

    train_dataset = MemeDatasetFromFile(path_train)  # Add your image transformations if needed
    train_dataloader = torch.utils.data.DataLoader \
        (train_dataset, batch_size=1, shuffle=False)

    test_dataset = MemeDatasetFromFile(path_test)  # Add your image transformations if needed
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    encoder = EncoderCNN(backbone='vgg16').to(device).eval()
    decoder = DecoderLSTM(hidden_size=512, output_size=train_dataset.n_words, num_layers=1).to(device).eval()
    load_checkpoint(decoder, "train_checkpoint/LSTM_Captions_ckpt_20.pth")

    wandb_logger = Logger(f"inm706_coursework_cnn_lstm_test", project='inm706_cw_inference', model=decoder)
    logger = wandb_logger.get_logger()

    test(test_dataloader, encoder, decoder, logger, test_dataset)
    return


if __name__ == '__main__':
    main()

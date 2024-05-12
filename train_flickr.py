import torch
import wandb
import nltk
import os

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from torchvision import transforms

from dataset_flickr import FlickrDataset
from logger import Logger
from models_captioning import EncoderCNN, DecoderLSTM
from utils_functions import save_checkpoint, load_checkpoint, load_config
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pack_padded_sequence



torch.manual_seed(0)

os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"  # Proxy to train with hyperion

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device set to {0}'.format(device))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def visualize_att(image, seq, alphas, mode="cap"):
    image = transforms.ToPILImage()(image[0].unsqueeze(0).squeeze(0))
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    # Only plot first element from batch
    caption = seq[0]
    words = caption
    for t in range(len(words)):
        if t > 50:
            break
        fig = plt.figure()
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[0, t, :]
        if current_alpha.size(0) == 196:
            current_alpha = current_alpha.view(-1, 14, 14).squeeze(0)
        else:
            current_alpha = current_alpha.view(-1, 14, 14).squeeze(0).mean(0)  # mean if object detection included

        alpha = skimage.transform.resize(current_alpha.cpu().numpy(), [14 * 24, 14 * 24])

        plt.imshow(alpha, alpha=0.65, cmap='Greys_r')
        plt.axis('off')
        if mode == "cap":
            wandb.log({"Caption Attention": wandb.Image(fig)})
        elif mode == "img":
            wandb.log({"Image Caption Attention": wandb.Image(fig)})
        # plt.show()
        plt.close(fig)

def visualize_encoder_attention(img, attention_map):
    """
    Visualize attention map on the image
    img: [3, W, H] PyTorch tensor (image)
    attention_map: [W*H, W*H] PyTorch tensor (attention map)
    """
    img = img.squeeze(0) # check shape
    attention_map = attention_map.squeeze(0)
    img = img.permute(1, 2, 0)  # [N, N, 3]
    attention_map = attention_map.cpu().detach().numpy()
    attention_map = np.mean(attention_map, axis=0)  # Average over first dimension

    attention_map = attention_map.reshape(int(np.sqrt(attention_map.shape[0])),
                                          int(np.sqrt(attention_map.shape[0])))  # Reshape to W * H
    attention_map = (attention_map - np.min(attention_map)) / (
                np.max(attention_map) - np.min(attention_map))  # Normalisation

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img.cpu().detach().numpy())
    plt.title('Image')

    plt.subplot(1, 2, 2)
    im = plt.imshow(attention_map, cmap='Greys_r')  # Add attention map
    plt.title('Attention Map')
    plt.colorbar(im, fraction=0.046, pad=0.04)  # Add colorbar as legend
    # plt.close(fig)
    wandb.log({"Encoder Attention": wandb.Image(fig)})


def evaluate(encoder, decoder_cap, input_tensor, caption, voc, mode="val", length=80, plot_encoder_attention=False):
    with torch.no_grad():
        if mode == "train":
            max_cap = length
            target_cap = caption
            plot_feature = False
        else:
            max_cap = length
            target_cap = None
            plot_feature = True

        encoder_outputs, weights = encoder(input_tensor, plot_feature)
        if plot_encoder_attention:
            visualize_encoder_attention(input_tensor[0], weights[0])
        caption_output, _, attention_weights = decoder_cap(encoder_outputs, caption,target_tensor=target_cap, max_caption=max_cap)
        caption_output = F.log_softmax(caption_output, dim=-1)
        decoded_caption = token2text(caption_output, voc)
    return decoded_caption, attention_weights


def calculate_bleu(target, predicted):
    references = [[caption] for caption in target]
    hypotheses = [caption for caption in predicted]
    # Calculate BLEU score
    bleu_score = nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                                                       smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
    return bleu_score


def calculate_meteor(target, predicted):
    references = [[caption] for caption in target]
    hypotheses = [caption for caption in predicted]

    # Calculate METEOR score
    meteor_score = nltk.translate.meteor_score.meteor_score(references, hypotheses)
    return meteor_score


def token2text(output, voc):
    EOS_token = 2
    _, topi = output.topk(1)
    decoded_ids = topi.squeeze(-1)
    decoded_words_total = []
    limit = decoded_ids.size(0)
    for i in range(limit):
        decoded_words = []
        caption = decoded_ids[i]

        for idx in caption:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(voc.index2word[idx.item()])
        decoded_words_total.append(decoded_words)
    return decoded_words_total


def target2text(target, voc):  # for BLEU EVALUATION
    EOS_token = 2
    decoded_words_total = []
    limit = target.size(0)
    for i in range(limit):
        decoded_words = []
        caption = target[i]

        for idx in caption:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(voc.index2word[idx.item()])
        decoded_words_total.append(decoded_words)
    return decoded_words_total


def val_epoch(dataloader, encoder, decoder_cap, criterion, output_lang, plot_encoder_attention=False,
              plot_decoder_attention=False):
    total_loss = 0
    total_samples = 0
    bleu_total = 0
    for data in dataloader:
        images = data["image"]
        meme_captions = data["meme_captions"].squeeze(1)
        max_caption = data["max_caption"]
        total_samples += 1
        with torch.no_grad():
            encoder_outputs, _ = encoder(images, False)
            caption_outputs, _, attention_weights = decoder_cap(encoder_outputs, meme_captions, None, max_caption)

        scores = pack_padded_sequence(caption_outputs, max_caption, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(meme_captions, max_caption, batch_first=True, enforce_sorted=False)[0]
        loss = criterion(scores, targets)

        hypothesis = target2text(meme_captions, output_lang)
        references = F.log_softmax(caption_outputs, dim=-1)
        references = token2text(references, output_lang)
        bleu_score = calculate_bleu(hypothesis, references)

        loss_attention = 0
        # loss_attention = 1.0 * ((1. - attention_weights.sum(dim=1)) ** 2).mean()

        loss_mix = loss + loss_attention
        bleu_loss = bleu_score

        if total_samples % 100 == 0:
            captions, attention_weights = evaluate(encoder, decoder_cap, images, meme_captions, output_lang, mode="val",
                                                   length=max_caption, plot_encoder_attention=plot_encoder_attention)
            if plot_decoder_attention:
                visualize_att(images, captions, attention_weights, mode="cap")
            total_text = ""
            for caption in captions:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_text += result
            print(f"DataPoint: {total_samples}, Text: {total_text}")

            total_target = ""
            targets_dec = target2text(meme_captions, output_lang)
            for caption in targets_dec:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_target += result
            print(f"Datapoint Target: {total_target}")

            print(f"Datapoint: {total_samples}, BLEU score: {bleu_score:.4f}")
            columns = ["Datapoint", "Output"]
            data = [[str(total_samples), total_text]]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"Validation example Caption": table})

        total_loss += loss_mix.item()
        bleu_total += bleu_loss

    return total_loss / len(dataloader), bleu_total / len(dataloader)


def train_epoch(dataloader, encoder, decoder_cap, encoder_optimizer, decoder_optimizer_cap, criterion, output_lang):
    total_loss = 0
    total_samples = 0
    for data in dataloader:
        images = data["image"]
        meme_captions = data["meme_captions"].squeeze(1)
        max_caption = data["max_caption"]
        encoder_outputs,_ = encoder(images)
        caption_outputs, _, attention_weights = decoder_cap(encoder_outputs, meme_captions,
                                                            meme_captions, max_caption)

        scores = pack_padded_sequence(caption_outputs, max_caption, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(meme_captions, max_caption, batch_first=True, enforce_sorted=False)[0]
        loss = criterion(scores, targets)
        loss_attention = 0

        # loss_mix = loss + 0.8*loss_img + 0.0001*l2_norm

        loss_cap = loss + loss_attention
        loss_mix = loss_cap

        if total_samples % 100 == 0:
            print(f"Datapoint: {total_samples}, loss_mix: {loss_mix.item()}, loss_attention: {loss_attention}")

        if torch.isnan(loss):
            print("NAN DETECTED")
            print(caption_outputs)
            nan_mask = torch.isnan(caption_outputs)
            print(nan_mask)

        decoder_optimizer_cap.zero_grad()
        encoder_optimizer.zero_grad()
        loss_mix.backward()
        grad_clip = 5.
        if grad_clip is not None:
            clip_gradient(decoder_optimizer_cap, grad_clip)
            clip_gradient(encoder_optimizer, grad_clip)
        decoder_optimizer_cap.step()
        encoder_optimizer.step()

        total_loss += loss_mix.item()
        total_samples += 1

    return total_loss / len(dataloader)


def train(train_dataloader, val_dataloader, encoder, decoder_cap, n_epochs, logger, output_lang,
          decoder_learning_rate=1e-4, encoder_learning_rate = 1e-5, print_every=100, plot_every=100,
          plot_encoder_attention=False, plot_decoder_attention=False):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    best_score = float('inf')

    val_plot_losses = []
    val_print_loss_total = 0  # Reset every print_every
    val_plot_loss_total = 0  # Reset every plot_every

    bleu_plot_losses = []
    bleu_print_loss_total = 0  # Reset every print_every
    bleu_plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_learning_rate)
    decoder_optimizer_cap = optim.Adam(decoder_cap.parameters(),lr=decoder_learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    for epoch in range(1, n_epochs + 1):
        print(epoch)
        encoder.train()
        decoder_cap.train()
        loss = train_epoch(train_dataloader, encoder, decoder_cap,encoder_optimizer, decoder_optimizer_cap,
                           criterion,output_lang)

        encoder.eval()
        decoder_cap.eval()
        val_loss, bleu_loss = val_epoch(val_dataloader, encoder, decoder_cap, criterion,output_lang, plot_encoder_attention=plot_encoder_attention, plot_decoder_attention=plot_decoder_attention)

        print_loss_total += loss
        plot_loss_total += loss

        val_print_loss_total += val_loss
        val_plot_loss_total += val_loss

        bleu_print_loss_total += bleu_loss
        bleu_plot_loss_total += bleu_loss

        if val_loss < best_score:
            save_checkpoint(decoder_cap, "Resnet-LSTM_Captions_decoder_Cap")
            save_checkpoint(encoder, "Resnet-LSTM_Captions_encoder")
            best_score = val_loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f"Train Epoch: {epoch}/{n_epochs}, Loss {print_loss_avg}")

            val_print_loss_avg = val_print_loss_total / print_every
            val_print_loss_total = 0
            print(f"Validation Epoch: {epoch}/{n_epochs}, Loss {val_print_loss_avg}")

            bleu_print_loss_avg = bleu_print_loss_total / print_every
            bleu_print_loss_total = 0
            print(f"Validation BLEU Epoch: {epoch}/{n_epochs}, Loss {bleu_print_loss_avg}")

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            logger.log({'train_loss_avg': plot_loss_avg})
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            val_plot_loss_avg = val_plot_loss_total / plot_every
            logger.log({'val_loss_avg': val_plot_loss_avg})
            val_plot_losses.append(val_plot_loss_avg)
            val_plot_loss_total = 0

            bleu_plot_loss_avg = bleu_plot_loss_total / plot_every
            logger.log({'val_bleu_avg': bleu_plot_loss_avg})
            bleu_plot_losses.append(bleu_plot_loss_avg)
            bleu_plot_loss_total = 0


def main():
    config = load_config()
    model_setting = config['model']
    train_setting = config['train']

    train_dataset = FlickrDataset()
    train_len = int(len(train_dataset) * 0.8)
    train_set, val_set = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    val_len = int(len(val_set) * 0.5)
    val_set, test_set = random_split(val_set, [val_len, len(val_set) - val_len])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_setting['batch_size'], shuffle=True)
    print("Training")
    print(len(train_set))
    print("Validation")
    print(len(val_set))
    print("Test")
    print(len(test_set))
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=train_setting['batch_size'], shuffle=True)

    encoder = EncoderCNN(backbone=model_setting['encoder_model_type'], attention=model_setting['encoder_attention']).to(device)
    decoder_cap = DecoderLSTM(hidden_size=512, embed_size=300, output_size=train_dataset.n_words, num_layers=1, attention=model_setting['decoder_bahdanau']).to(
        device)

    wandb_logger = Logger(f"FLICKR-resnet-baseline",
                          project='INM706-FINAL', model=decoder_cap)
    logger = wandb_logger.get_logger()

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    print("\n############## TRAIN SETTINGS ##############")
    print(train_setting)
    print()
    n_epochs = train_setting['epochs']
    print(f"Length of vocabulary: {train_dataset.n_words}")

    train(train_dataloader, val_dataloader, encoder, decoder_cap, n_epochs, logger, train_dataset,
          print_every=1,plot_every=1, encoder_learning_rate=train_setting['encoder_learning_rate'], decoder_learning_rate=train_setting['decoder_learning_rate'],
          plot_encoder_attention=model_setting['encoder_attention'], plot_decoder_attention=model_setting['decoder_bahdanau'])
    return


# TODO:
# Baseline: Resnet vs EfficientNet
# Improvement: EfficientNet with Bahdanau attention
# Evaluate: Add self attention to encoder


if __name__ == '__main__':
    main()

import time
import torch
import wandb
import nltk
import os

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import skimage.transform
import matplotlib.cm as cm
from torchvision import transforms

from dataset import MemeDatasetFromFile
from dataset_flickr import FlickrDataset
from logger import Logger
from models_captioning import EncoderCNN, DecoderLSTM
from utils_functions import save_checkpoint, load_checkpoint
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn.functional import cosine_similarity
from torch.nn.utils.rnn import pack_padded_sequence
# from gensim.models import Word2Vec



torch.manual_seed(0)

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"  # Proxy to train with hyperion

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


def visualize_att(image, seq, alphas, smooth=False, mode="cap"):
    image = transforms.ToPILImage()(image[0].unsqueeze(0).squeeze(0))
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    # Only plot first element from batch
    caption = seq[0]
    # print(caption)
    words = caption
    # print(alphas.shape)
    # alphas = alphas[:, :-1]  # remove last weight for resize/visualisation reasons
    for t in range(len(words)):
        if t > 50:
            break
        # plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)
        fig = plt.figure()
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[0,t, :]
        # print(current_alpha.shape)
        if current_alpha.size(0) == 196:
            current_alpha = current_alpha.view(-1, 14, 14).squeeze(0) # mean if object detections
        else:
            current_alpha = current_alpha.view(-1, 14, 14).squeeze(0).mean(0) # mean if object detectio  included

        # alpha = np.reshape(current_alpha, (224, 224))  # Resize to image dimensions
        # print(current_alpha.shape)
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.cpu().numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.cpu().numpy(), [14 * 24, 14 * 24])

        plt.imshow(alpha, alpha=0.65, cmap='Greys_r')
        plt.axis('off')
        if mode == "cap":
            wandb.log({"Caption Attention": wandb.Image(fig)})
        elif mode == "img":
            wandb.log({"Image Caption Attention": wandb.Image(fig)})
        # plt.show()
        plt.close(fig)


def evaluate(encoder, decoder_cap, input_tensor, caption, voc, mode="val", length = 80):
    with torch.no_grad():
        # print(length)
        if mode == "train":
            max_cap = length
            target_cap = caption
            plot_feature = True
        else:
            max_cap = length
            target_cap = None
            plot_feature = False

        encoder_outputs = encoder(input_tensor, plot_feature)
        caption_output, _, attention_weights = decoder_cap(encoder_outputs, caption,
                                                           target_tensor=target_cap, max_caption=max_cap)
        caption_output = F.log_softmax(caption_output, dim=-1)
        decoded_caption = token2text(caption_output, voc)
    return decoded_caption, attention_weights


def calculate_bleu(target, predicted):

    references = [[caption] for caption in target]
    hypotheses = [caption for caption in predicted]
    # Calculate BLEU score
    bleu_score = nltk.translate.bleu_score.corpus_bleu(references, hypotheses,weights=(0.25, 0.25, 0.25, 0.25),
                                                       smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
    return bleu_score

# def calculate_cosine(target_captions, predicted_captions):
#     model = Word2Vec.load("path/to/word2vec_model")
#     predicted_vecs = []
#     for caption in predicted_captions:
#         vec = [model.wv[word] for word in caption.split() if word in model.wv]
#         predicted_vecs.append(vec)
#
#     target_vecs = []
#     for caption in target_captions:
#         vec = [model.wv[word] for word in caption.split() if word in model.wv]
#         target_vecs.append(vec)
#
#     # Calculate cosine similarity pairwise between predicted and target vectors
#     cosine_similarity = []
#     for i in range(len(predicted_captions)):
#         for j in range(len(target_captions)):
#             similarity = model.wv.cosine_similarities(predicted_vecs[i], target_vecs[j])[0]
#             cosine_similarity.append(similarity)
#             # print(
#             #     f"Similarity between predicted '{predicted_captions[i]}' and target '{target_captions[j]}': {similarity:.4f}"
#     return sum(cosine_similarity)/len(cosine_similarity)

def token2text(output, voc):
    EOS_token = 2
    UNK = 3
    _, topi = output.topk(1)
    decoded_ids = topi.squeeze(-1)
    # print(decoded_ids)
    # print(decoded_ids.shape)
    decoded_words_total = []
    limit = decoded_ids.size(0)
    # print(limit)
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

def target2text(target, voc): # for BLEU EVALUATION
    EOS_token = 2
    UNK = 3
    decoded_words_total = []
    limit = target.size(0)
    # print(limit)
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


def val_epoch(dataloader, encoder, decoder_cap, decoder_img, criterion, output_lang):
    total_loss = 0
    total_samples = 0
    bleu_total = 0
    for data in dataloader:
        images = data["image"]
        meme_captions = data["meme_captions"].squeeze(1)
        max_caption = data["max_caption"]
        total_samples += 1
        img_captions = data["img_captions"]
        with torch.no_grad():
            encoder_outputs = encoder(images, False)
            caption_outputs, _, attention_weights = decoder_cap(encoder_outputs, meme_captions,
                                                                None, max_caption)


        scores = pack_padded_sequence(caption_outputs, max_caption, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(meme_captions, max_caption, batch_first=True, enforce_sorted=False)[0]
        loss = criterion(scores, targets)

        hypothesis = target2text(meme_captions, output_lang)
        references = F.log_softmax(caption_outputs, dim=-1)
        references = token2text(references, output_lang)
        bleu_score = calculate_bleu(hypothesis, references)

        loss_attention = 0
        # loss_attention = 1.0 * ((1. - attention_weights.sum(dim=1)) ** 2).mean()

        l2_norm = sum(torch.norm(p, 2) ** 2 for p in decoder_cap.parameters())
        loss_mix = loss + loss_attention
        bleu_loss = bleu_score

        if total_samples % 100 == 0:
            captions, attention_weights = evaluate(encoder, decoder_cap, images,meme_captions,
                                                                                       output_lang, mode="val",length = max_caption)
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

    return total_loss / len(dataloader), bleu_total/len(dataloader)


def train_epoch(dataloader, encoder, decoder_cap, decoder_img, encoder_optimizer,
                decoder_optimizer_cap, decoder_optimizer_img, criterion, output_lang, logger):
    total_loss = 0
    batch_loss = 0.0
    total_samples = 0
    for data in dataloader:
        # print(f"Datapoint {total_samples}")
        images = data["image"]
        titles = data["title"]
        meme_captions = data["meme_captions"].squeeze(1)
        img_captions = data["img_captions"]
        max_caption = data["max_caption"]
        max_img = data["max_img"]
        encoder_outputs = encoder(images)
        # print(encoder_outputs.shape)
        caption_outputs, _, attention_weights = decoder_cap(encoder_outputs, meme_captions,
                                                            meme_captions, max_caption)

        # scores = caption_outputs.view(-1, caption_outputs.size(-1))
        # targets = meme_captions[:,:max(max_caption)]
        # print(max_caption)
        scores = pack_padded_sequence(caption_outputs, max_caption, batch_first=True, enforce_sorted = False)[0]
        # print(pack_padded_sequence(caption_outputs, max_caption, batch_first=True, enforce_sorted = False)[1])
        targets = pack_padded_sequence(meme_captions, max_caption, batch_first=True, enforce_sorted = False)[0]
        # print(pack_padded_sequence(meme_captions, max_caption, batch_first=True, enforce_sorted = False)[1])
        loss = criterion(scores, targets)
        loss_attention = 0

        l2_norm = sum(torch.norm(p, 2) ** 2 for p in decoder_cap.parameters())

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
        batch_loss += loss_mix.item()

        total_samples += 1
        if total_samples % 500 == 0:
            total_text = ""
            captions, attention_weights = evaluate(encoder, decoder_cap, images,meme_captions,output_lang, mode="train", length = max_caption)

            visualize_att(images, captions, attention_weights, mode="cap")

            for caption in captions:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_text += result
            print(f"Datapoint Attention: {total_samples}, Text: {total_text}")

            total_target = ""
            targets_dec = target2text(meme_captions, output_lang)
            for caption in targets_dec:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_target += result
            print(f"Datapoint Attention Target: {total_target}")

            bleu_score = calculate_bleu(targets_dec, captions)
            print(f"Datapoint Attention: {total_samples}, BLEU score: {bleu_score:.4f}")

            columns = ["Datapoint", "Output"]
            data = [[str(total_samples), total_text]]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"Train example Caption": table})

    return total_loss / len(dataloader)


def train(
        train_dataloader, val_dataloader, encoder, decoder_cap, decoder_img, n_epochs, logger, output_lang,
        learning_rate=4e-4,
        print_every=100, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    val_plot_losses = []
    val_print_loss_total = 0  # Reset every print_every
    val_plot_loss_total = 0  # Reset every plot_every

    bleu_plot_losses = []
    bleu_print_loss_total = 0  # Reset every print_every
    bleu_plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=1e-5)
    decoder_optimizer_cap = optim.Adam(params=filter(lambda p: p.requires_grad, decoder_cap.parameters()), lr=learning_rate)
    decoder_optimizer_img = optim.Adam(decoder_img.parameters(), lr=learning_rate)
    # criterion = nn.NLLLoss(ignore_index=0)
    # criterion = nn.NLLLoss(ignore_index=0).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)


    for epoch in range(1, n_epochs + 1):
        print(epoch)
        loss = train_epoch(train_dataloader, encoder.train(), decoder_cap.train(), decoder_img.train(),
                           encoder_optimizer, decoder_optimizer_cap, decoder_optimizer_img,
                           criterion,
                           output_lang, logger)

        val_loss, bleu_loss = val_epoch(val_dataloader, encoder.eval(), decoder_cap.eval(), decoder_img.eval(), criterion,
                             output_lang)

        print_loss_total += loss
        plot_loss_total += loss

        val_print_loss_total += val_loss
        val_plot_loss_total += val_loss

        bleu_print_loss_total += bleu_loss
        bleu_plot_loss_total += bleu_loss

        if epoch % print_every == 0:
            if epoch % 5 == 0:
                save_checkpoint(epoch, decoder_cap, "LSTM_Captions_decoder_Cap", decoder_optimizer_cap)
                save_checkpoint(epoch, encoder, "LSTM_Captions_encoder", encoder_optimizer)
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
    n_epochs = 40

    path_test = "data/memes-test.json"
    path_train = "data/memes-trainval.json"

    train_dataset = FlickrDataset(path_train)
    train_len = int(len(train_dataset) * 0.9)
    train_set, val_set = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=False)
    print("Training")
    print(len(train_set))
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)

    encoder = EncoderCNN(backbone='efficientnet').to(device)
    decoder_cap = DecoderLSTM(hidden_size=512, embed_size=300, output_size=train_dataset.n_words, num_layers=1).to(
        device)
    decoder_img = DecoderLSTM(hidden_size=512, embed_size=300, output_size=train_dataset.n_words, num_layers=1).to(
        device)

    wandb_logger = Logger(f"inm706_coursework_cnn_lstm_resnet_attention_flickr_exp",
                          project='inm706_cw_hyperion_resnet_attention_flickr_exp', model=decoder_cap)
    logger = wandb_logger.get_logger()

    print(f"Length of vocabulary: {train_dataset.n_words}")

    train(train_dataloader, val_dataloader, encoder, decoder_cap, decoder_img, n_epochs, logger, train_dataset,
          print_every=1,
          plot_every=1)
    return


if __name__ == '__main__':
    main()

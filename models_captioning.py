import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import random
import wandb
import matplotlib.pyplot as plt

#from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel


from efficientnet_pytorch import EfficientNet
import matplotlib.patches as patches
from torchvision import transforms

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


# TODO:
# RESNET 101 as in paper, 2048, some layers fine tuned
# Self attention on image and on joined sequence?

class LearnedPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LearnedPooling, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x is of shape (batch_size, sequence_length, input_dim)
        # Apply the linear layer to each sequence element
        x = self.linear(x)
        # Now x is of shape (batch_size, sequence_length, output_dim)
        # Apply a softmax over the sequence dimension to get the weights
        weights = torch.nn.functional.softmax(x, dim=1)
        # Multiply the original sequence by the weights and sum over the sequence dimension
        pooled = (x * weights).sum(dim=1)
        # Now pooled is of shape (batch_size, output_dim), and can be used as the initial hidden state of the LSTM
        return pooled


class EncoderCNN(nn.Module):
    def __init__(self, backbone='vgg16'):
        super(EncoderCNN, self).__init__()


        # self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        # torch.save(self.efficientnet.state_dict(), 'checkpoint/efficient_weights.pth')
        # self.efficientnet._avg_pooling = nn.Identity()
        # self.efficientnet._fc = nn.Identity()
        # for param in self.efficientnet.parameters():
        #     param.requires_grad = False
        #     for param in list(self.efficientnet.parameters())[-2:]:
        #         param.requires_grad = True
        # self.features = self.efficientnet

        # self.cnn_model = torchvision.models.vgg16(pretrained=True)
        # torch.save(self.cnn_model.state_dict(), 'checkpoint/model_weights.pth')
        # self.cnn_model = torchvision.models.vgg16()  # we do not specify ``weights``, i.e. create untrained model
        # self.cnn_model.load_state_dict(torch.load("checkpoint/model_weights.pth"))
        # self.cnn_model.eval()

        # self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        # self.faster_rcnn.load_state_dict(torch.load("checkpoint/faster_rcnn_model_weights.pth"))
        # self.faster_rcnn.eval()

        # for name, p in self.faster_rcnn.named_parameters():
        #     p.requires_grad = False

        # self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # torch.save(self.yolo.state_dict(), 'checkpoint/yolo_weights.pth')
        # self.yolo.eval()

        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        if backbone == "resnet":
            self.resnet = torchvision.models.resnet101()
            # torch.save(self.resnet.state_dict(), 'checkpoint/resnet_weights.pth')
            self.resnet.load_state_dict(torch.load("checkpoint/resnet_weights.pth"))
            self.features = nn.Sequential(*list(self.resnet.children())[:-2])
            print(list(self.features.children()))

            for p in self.features.parameters():
                p.requires_grad = False
                # If fine-tuning, only fine-tune convolutional blocks 2 through 4
            for c in list(self.features.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = True

        if backbone == "efficientnet":
            self.model = EfficientNet.from_pretrained('efficientnet-b0') # Load a pretrained EfficientNet model
            self.model._avg_pooling = nn.Identity()
            self.model._dropout = nn.Identity()
            num_features = self.model._fc.in_features
            print(num_features)
            self.model._fc = nn.Identity()


        # if backbone == "clip":
        #     # model_name = "openai/clip-vit-base-patch32"
        #     model_name = "openai/clip-vit-base-patch16"
        #     # self.clip = CLIPModel.from_pretrained(model_name)
        #     self.clip = CLIPVisionModel.from_pretrained(model_name)
        #     # processing = CLIPProcessor.from_pretrained(model_name)


        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        # print(list(self.features.parameters()))

        # torch.save(self.features.state_dict(), 'checkpoint/resnet_weights.pth')
        # self.linear_embed = nn.Linear(5632, 512) # 41472, 32768

        # self.learn_pool = LearnedPooling(512,512)
        # self.conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1) (1, 131072)
        # self.attention = AttentionMultiHead(2048, 2048,4)

        # self.avgpool = self.cnn_model.avgpool

    def forward(self, input, plot_features=False):
        # print("Input")
        # print(input.shape)
        # x = self.features(input[0].unsqueeze(0))
        x = self.model.extract_features(input[0].unsqueeze(0))

        # with torch.no_grad():
            # x = self.clip.get_image_features(pixel_values=input)
            # x,y = self.clip(pixel_values=input, return_dict=False)
            # print(y.shape)
            # print(x.shape)

        # print(x.shape)
        if plot_features:
            np_image = input[0].unsqueeze(0).squeeze().permute(1, 2, 0).cpu().numpy()
            fig = plt.figure()
            plt.imshow(np_image)
            wandb.log({"Extracted features": wandb.Image(fig)})
            plt.close(fig)

            fig = plt.figure()
            # # Visualize the feature map
            feature_map_np = x.mean(1).detach().cpu().numpy()
            # print(feature_map_np.shape)
            for i in range(x.size(0)):
                plt.imshow(feature_map_np[i], cmap='viridis')  # Choose a suitable colormap
                plt.title("Feature Map from CNN Layer")
                plt.axis('off')  # Hide axes
                wandb.log({"Extracted features": wandb.Image(fig)})
                plt.close(fig)

        # self.faster_rcnn.eval()
        # with torch.no_grad():
        #     object_features = self.faster_rcnn(input)
        # boxes = object_features[0]['boxes'][:5]
        # labels = object_features[0]['labels'][:5]
        # # print(len(labels))
        #
        # # if len(labels)>0:
        # #     fig, ax = plt.subplots(1)
        # #     np_image = input.squeeze().permute(1, 2, 0).cpu().numpy()
        # #     np_boxes = boxes.cpu().numpy()
        # #     ax.imshow(np_image)
        # #     for box, label in zip(np_boxes, labels):
        # #         rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
        # #                                  facecolor='none')
        # #         ax.add_patch(rect)
        # #         plt.text(box[0], box[1], self.COCO_INSTANCE_CATEGORY_NAMES[label], fontsize=10,
        # #                  bbox=dict(facecolor='yellow', alpha=0.5))
        # #     plt.show()
        #
        # # Extract features for each object detected
        object_features = []
        # if len(labels) > 0:
        #     for box in boxes:
        #         cropped_image = input[:, :, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        #         padded_image = torch.zeros_like(input)
        #         padded_image[:, :, int(box[1]):int(box[3]), int(box[0]):int(box[2])] = cropped_image
        #         # print(padded_image.shape)
        #         # padding = torch.nn.ConstantPad2d(
        #         #     (0, input.shape[3] - object_image.shape[3], 0, input.shape[2] - object_image.shape[2]), 0)
        #         # object_image = padding(object_image)
        #         # object_image = torch.nn.functional.interpolate(object_image, size=input.shape[2:])
        #         # with torch.no_grad():
        #         #     object_feature = self.faster_rcnn.backbone(padded_image)
        #         object_feature = self.features(padded_image)
        #
        #         # np_image = padded_image.squeeze().permute(1, 2, 0).cpu().numpy()
        #         # plt.imshow(np_image)
        #         # plt.show()
        #         # # # Visualize the feature map
        #         # feature_map_np = object_feature.mean(1).detach().cpu().numpy()
        #         # # print(feature_map_np.shape)
        #         # for i in range(object_feature.size(0)):
        #         #     plt.imshow(feature_map_np[i], cmap='viridis')  # Choose a suitable colormap
        #         #     plt.title("Feature Map from CNN Layer")
        #         #     plt.axis('off')  # Hide axes
        #         #     plt.show()
        #
        #         object_feature = self.adaptive_pool(
        #             object_feature)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        #         object_feature = object_feature.permute(0, 2, 3, 1)
        #         object_feature = object_feature.contiguous().view(object_feature.size(0), -1,
        #                                                                       object_feature.size(-1))
        #         # print(object_feature.shape)
        #         object_features.append(object_feature)
        # else:  # if no object divide image into equal parts
        #     # print("No detection!")
        #     rows = torch.split(input, input.shape[3] // 3, dim=2)
        #     # print("Rows")
        #     # print(len(rows))
        #     for row in rows:
        #         # Split each row into 3 parts along the width dimension
        #         parts = torch.split(row, input.shape[2] // 3, dim=3)
        #         # print("Parts")
        #         # print(len(parts))
        #         for part in parts:
        #             padding = torch.nn.ConstantPad2d(
        #                 (0, input.shape[3] - part.shape[3], 0, input.shape[2] - part.shape[2]), 0)
        #             object_image = padding(part)
        #             # object_image = torch.nn.functional.interpolate(object_image, size=input.shape[2:])
        #             # print(object_image.shape)
        #             object_feature = self.features(object_image)
        #             object_feature = self.adaptive_pool(
        #                 object_feature)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        #             # object_feature = object_feature.mean(dim=[2, 3]).unsqueeze(1)
        #             object_feature = object_feature.view(1, 2048, -1).transpose(1, 2)
        #             object_features.append(object_feature)
        # print("CONCAT PART")
        # whole_image_features = self.features(input)
        whole_image_features = self.model.extract_features(input)
        whole_image_features = self.adaptive_pool(
            whole_image_features)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        whole_image_features = whole_image_features.permute(0, 2, 3, 1)
        whole_image_features = whole_image_features.contiguous().view(whole_image_features.size(0), -1,  whole_image_features.size(-1))

        # with torch.no_grad():
        # whole_image_features,_ = self.clip(pixel_values=input, return_dict=False)

        # whole_image_features = whole_image_features.mean(dim=[2, 3]).unsqueeze(1)

        # print(whole_image_features.shape)

        # max_len = max(len(whole_image_features), max([len(x) for x in object_features]))
        # print(max_len)
        # whole_image_features = torch.nn.functional.pad(whole_image_features, (0, max_len - len(whole_image_features)))
        # object_features = [torch.nn.functional.pad(x, (0, max_len - len(x))) for x in object_features]

        # print(f"Image Max: {torch.max(whole_image_features)}, Min: {torch.min(whole_image_features)}")
        # features = torch.cat([whole_image_features] + object_features, dim=1)
        return whole_image_features


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(1280, hidden_size)  # 2048 for resnet101
        self.Va = nn.Linear(hidden_size, 1)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, query, keys):
        if type(query) is tuple:
            query, _ = query

        # print(query.shape)
        # print(keys.shape)
        hidden = self.W1(query)
        encoder = self.W2(keys)
        # print(hidden.shape)
        # print(encoder.shape)
        scores = self.Va(torch.relu(hidden.unsqueeze(1) + encoder)).squeeze(2)
        # print(scores.shape)
        weights = F.softmax(scores, dim=1)
        context = (keys * weights.unsqueeze(2)).sum(dim=1)
        return context, weights


class AttentionMultiHead(nn.Module):

    def __init__(self, input_size, hidden_size, nr_heads):
        super(AttentionMultiHead, self).__init__()
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList([])
        self.heads.extend([SelfAttention(input_size, hidden_size) for idx_head in range(nr_heads)])
        self.linear_out = nn.Linear(nr_heads * hidden_size, input_size)
        return

    def forward(self, input_vector):
        all_heads = []
        for head in self.heads:
            out = head(input_vector)
            all_heads.append(out)
        z_out_concat = torch.cat(all_heads, dim=2)
        z_out_out = F.relu(self.linear_out(z_out_concat))
        # print(z_out_out.shape)
        return z_out_out


class SelfAttention(nn.Module):

    def __init__(self, input_size, out_size):
        super(SelfAttention, self).__init__()
        self.dk_size = out_size
        self.query_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.key_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.value_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.softmax = nn.Softmax()
        self.apply(self.init_weights)
        return

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input_vector):
        query_out = F.relu(self.query_linear(input_vector))
        key_out = F.relu(self.key_linear(input_vector))

        value_out = F.relu(self.value_linear(input_vector))
        # out_q_k = torch.bmm(query_out, key_out.transpose(1, 2))
        out_q_k = torch.div(torch.bmm(query_out, key_out.transpose(1, 2)), math.sqrt(self.dk_size))
        softmax_q_k = self.softmax(out_q_k)
        out_combine = torch.bmm(softmax_q_k, value_out)
        return out_combine


class DecoderLSTM(nn.Module):
    MAX_LENGTH = 80
    SOS_token = 1
    EOS_token = 2

    def __init__(self, hidden_size, embed_size, output_size, num_layers=1, dropout_p=0.1, vocab=None):
        super(DecoderLSTM, self).__init__()
        if vocab is None:
            self.embedding = nn.Embedding(output_size, embed_size)
        # else:
        #     glove = GloVe(name='6B', dim=300)
        #     self.embedding = nn.Embedding(output_size, 300)
        #     print(output_size)
        #     print(len(vocab.index2word))
        #     # print(vocab.index2word)
        #
        #     # Initialize the embedding layer with the GloVe embedding weights
        #     not_glove = 0
        #     for i, word in enumerate(vocab.index2word):  # itos: index-to-string
        #         # print(word)
        #         word = vocab.index2word[i]
        #         # print(word)
        #         if word in glove.stoi:  # stoi: string-to-index
        #             self.embedding.weight.data[i] = glove[word]
        #         else:
        #             # print(word)
        #             not_glove += 1
        #             # For words not in the pretrained vocab, initialized them randomly
        #             self.embedding.weight.data[i] = torch.randn(300)
        #     print(f"Not in GLOVE: {not_glove}")
        #     self.embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(0.5)
        self.LSTM = nn.LSTMCell(embed_size + 1280, hidden_size)
        # self.LSTM = nn.LSTM(embed_size + 2048, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)  # 2 for direction and 2 for image/hidden concat
        self.num_layers = num_layers
        self.attention = BahdanauAttention(hidden_size)
        # self.attention = SelfAttention(hidden_size)
        # self.attention_function = self.forward_step
        self.attention_function = self.forward_step_bahdanau
        # self.attention_function = self.forward_step_self
        self.init_h = nn.Linear(1280, hidden_size)  # linear layer to find initial hidden state of LSTM
        self.init_c = nn.Linear(1280, hidden_size)  # linear layer to find initial cell state of LSTM
        # self.bn = nn.BatchNorm1d(hidden_size)

        self.s_gate = nn.Linear(hidden_size, 1280)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        ####################################################
        # init weights
        self.apply(self.init_weights)
        # self.attention_features = AttentionMultiHead(2048, 2048,4)


        # total_weights = 0
        # for x in self.named_parameters():
        #     if 'weight' in x[0]:
        #         if 'batch' not in x[0]:
        #             torch.nn.init.xavier_uniform_(x[1])
        #     elif 'bias' in x[0]:
        #         x[1].data.fill_(0.01)
        #     total_weights += x[1].numel()
        # print("A total of {0:d} parameters in LSTM".format(total_weights))

        # self.embedding.weight.data.uniform_(-0.1, 0.1)
        # self.out.bias.data.fill_(0)
        # self.out.weight.data.uniform_(-0.1, 0.1)
        ####################################################

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, feature_outputs, caption, target_tensor=None, max_caption=80):
        # target_tensor = None
        # target_tensor_img = None
        # print(target_tensor.shape)
        # print(max_caption)
        # print(target_tensor_img.shape)
        # feature_outputs = self.attention_features(feature_outputs)
        caption_outputs, hidden, attention = self.generate_caption(feature_outputs, caption, target_tensor,
                                                                   max_length=max_caption)
        return caption_outputs, hidden, attention

    def generate_caption(self, feature_outputs, caption, target_tensor=None, max_length=80):
        decoder_outputs = []
        attentions = []
        batch_size = caption.size(0)

        max_length = max(max_length)
        # print(max_length)
        # print(num_captions)
        target_embed = self.embedding(caption)
        # print(target_embed.shape)
        decoder_input_start = torch.empty(batch_size, dtype=torch.long, device=device).fill_(self.SOS_token)
        decoder_input = self.embedding(decoder_input_start)
        # print(decoder_input.shape)

        # feature_outputs = feature_outputs.repeat(1, num_captions, 1)  # num_layers and 2 for direction

        decoder_hidden = feature_outputs

        for i in range(max_length):
            if i == 0:
                decoder_output, decoder_hidden, attn_weights = self.attention_function(decoder_input, decoder_hidden,
                                                                                       feature_outputs, True)
            else:
                decoder_output, decoder_hidden, attn_weights = self.attention_function(decoder_input, decoder_hidden,
                                                                                       feature_outputs, False)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            if target_tensor is not None:
                probability = random.random()
                if probability < 0.5:
                    decoder_input = target_embed[:, i,:]  # Teacher forcing
                else:
                    # _, topi = decoder_output.topk(1)
                    # indices = torch.randint(1, (batch_size, 1)).to(device)
                    # decoder_input = torch.gather(topi, dim=1, index=indices).squeeze(-1).detach()
                    # decoder_input = self.embedding(decoder_input)
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(-1).detach()
                    decoder_input = self.embedding(decoder_input)
                    # print(decoder_input.shape)


            else:  # validation/no teacher forcing
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
                decoder_input = self.embedding(decoder_input)
                # _, topi = decoder_output.topk(1)
                # indices = torch.randint(1, (batch_size, 1)).to(device)
                # decoder_input = torch.gather(topi, dim=1, index=indices).squeeze(-1).detach()
                # decoder_input = self.embedding(decoder_input)

                # decoder_input = topi.squeeze(-1).detach()  # detach from history as input
                # print(decoder_input.shape)
        # print(decoder_outputs[0].shape)
        decoder_outputs = torch.cat([tensor.unsqueeze(1) for tensor in decoder_outputs], dim=1)
        # print(decoder_outputs.shape)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        if attentions[0] is not None:
            attentions = torch.cat([tensor.unsqueeze(1) for tensor in attentions], dim=1)
        else:
            attentions = None
        # print(attentions.shape)
        return decoder_outputs, decoder_hidden, attentions  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden, image, image_feature):
        mean_encoder_out = image.mean(dim=1)
        if type(hidden) is tuple:
            hidden, cell = hidden
        if image_feature:
            hidden = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
            cell = self.init_c(mean_encoder_out)
            # cell = torch.zeros_like(hidden)
            # hidden = torch.zeros_like(hidden)

        # print("Forward")
        # print(context.shape)
        # print(input.shape)
        # print(mean_encoder_out.shape)
        # print(context.shape)
        # print(input_lstm.shape)
        hidden, cell = self.LSTM(input, (hidden, cell))
        output = self.out(self.dropout(hidden))
        return output, (hidden, cell), output

    def forward_step_bahdanau(self, input, hidden, image, image_feature):
        mean_encoder_out = image.mean(dim=1)
        if type(hidden) is tuple:
            hidden, cell = hidden
        if image_feature:
            hidden = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
            cell = self.init_c(mean_encoder_out)
            # cell = torch.zeros_like(hidden)
            # hidden = torch.zeros_like(hidden)

        context, attn_weights = self.attention(hidden, image)
        # print("Forward")
        # print(context.shape)
        # print(input.shape)
        # print(mean_encoder_out.shape)
        gate = self.sigmoid(self.s_gate(hidden))
        context = context * gate
        # print(context.shape)
        input_lstm = torch.cat((input, context), dim=1)
        # print(input_lstm.shape)
        hidden, cell = self.LSTM(input_lstm, (hidden, cell))
        output = self.out(self.dropout(hidden))
        return output, (hidden,cell), attn_weights

    def forward_step_self(self, input, hidden, image, image_feature, mode):
        input = self.dropout(self.embedding(input))
        input = F.relu(input)

        mean_encoder_out = image.mean(dim=1).unsqueeze(1)
        mean_encoder_out = mean_encoder_out.repeat(self.num_layers, 1, 1)  # num_layers and 2 for direction
        if type(hidden) is tuple:
            hidden, cell = hidden
        if image_feature:
            hidden = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
            cell = self.init_c(mean_encoder_out)
        if image_feature:
            output, hidden = self.LSTM(input, (hidden, hidden))
        else:
            output, hidden = self.LSTM(input, hidden)
            # print("Hidden shape")
            # print(hidden[0].shape)
        # print(output.shape)
        context = self.attention(output)
        # print(context.shape)
        # print(output.shape)

        # Concatenate decoder hidden state and context vector
        combined = torch.cat((output, context), dim=2)
        # print(combined.shape)
        # print(combined.shape)
        if mode == "img":
            # print("Image")
            output = self.out_img(combined)
        else:
            output = self.out(combined)

        return output, hidden, context

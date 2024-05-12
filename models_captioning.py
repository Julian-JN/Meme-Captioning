import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import random
import wandb
import matplotlib.pyplot as plt


from efficientnet_pytorch import EfficientNet


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class EncoderCNN(nn.Module):
    def __init__(self, backbone='resnet', attention = True):
        super(EncoderCNN, self).__init__()

        self.model_type = backbone
        self.multihead = attention

        if backbone == "resnet":
            self.model = torchvision.models.resnet101(pretrained=True)
            # torch.save(self.resnet.state_dict(), 'checkpoint/resnet_weights.pth')
            self.model.load_state_dict(torch.load("checkpoint/resnet_weights.pth"))
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
            num_features = 2048

        if backbone == "efficientnet":
            self.model = EfficientNet.from_pretrained('efficientnet-b0') # Load a pretrained EfficientNet model
            self.model._avg_pooling = nn.Identity()
            self.model._dropout = nn.Identity()
            num_features = 1280
            print(num_features)
            self.model._fc = nn.Identity()
        if attention:
            self.attention = AttentionMultiHeadCNN(input_size=num_features, hidden_size=num_features,nr_heads=4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, input, plot_features=False):

        if plot_features:
            if self.model_type == "resnet":
                x = self.model(input[0].unsqueeze(0))
            else:
                x = self.model.extract_features(input[0].unsqueeze(0))
            # Retrieve the original image and overlay extracted feature maps over it
            np_image = input[0].unsqueeze(0).squeeze().permute(1, 2, 0).cpu().numpy()
            fig = plt.figure()
            plt.imshow(np_image)
            plt.close(fig)
            fig = plt.figure()
            feature_map_np = x.mean(1).detach().cpu().numpy()
            for i in range(x.size(0)):
                plt.imshow(feature_map_np[i], cmap='viridis')  # Choose a suitable colormap
                plt.title("Feature Map from CNN Layer")
                plt.axis('off')  # Hide axes
                wandb.log({"Extracted features": wandb.Image(fig)})
                plt.close(fig)

        if self.model_type == "resnet":
            whole_image_features = self.model(input)
        else:
            whole_image_features = self.model.extract_features(input) # (batch_size, 2048, encoded_image_size, encoded_image_size)
        weights = 0
        if self.multihead:
            context, weights = self.attention(whole_image_features) # (batch_size, 2048, encoded_image_size, encoded_image_size)
        whole_image_features = self.adaptive_pool(whole_image_features)  # (batch_size, 2048, 14, 14)
        whole_image_features = whole_image_features.permute(0, 2, 3, 1)
        whole_image_features = whole_image_features.contiguous().view(whole_image_features.size(0), -1,  whole_image_features.size(-1)) # (batch_size, 2048, 196)
        return whole_image_features, weights

class AttentionMultiHeadCNN(nn.Module):

    def __init__(self, input_size, hidden_size, nr_heads):
        super(AttentionMultiHeadCNN, self).__init__()
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList([])
        self.heads.extend([SelfAttentionCNN(input_size) for idx_head in range(nr_heads)])
        self.conv_out = nn.Conv2d(in_channels=input_size*nr_heads, out_channels=input_size, kernel_size=1)
        return

    def forward(self, input_vector):
        all_heads = []
        all_weights = []
        for head in self.heads:
            out, weights = head(input_vector)
            all_heads.append(out)
            all_weights.append(weights)
        z_out_concat = torch.cat(all_heads, dim=1)
        weight_mean = torch.mean(torch.stack(all_weights), dim=0)
        z_out_out = F.relu(self.conv_out(z_out_concat))
        # print(z_out_out.shape)
        return z_out_out, weight_mean

class SelfAttentionCNN(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttentionCNN, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        attention_temp = torch.bmm(attention, attention.permute(0,2,1))
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out_att = out.view(batch_size, C, width, height)
        out = self.gamma * out_att + x
        return out, attention_temp


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
        hidden = self.W1(query)
        encoder = self.W2(keys)
        scores = self.Va(torch.relu(hidden.unsqueeze(1) + encoder)).squeeze(2)
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
        out_q_k = torch.div(torch.bmm(query_out, key_out.transpose(1, 2)), math.sqrt(self.dk_size))
        softmax_q_k = self.softmax(out_q_k)
        out_combine = torch.bmm(softmax_q_k, value_out)
        return out_combine


class DecoderLSTM(nn.Module):
    MAX_LENGTH = 80
    SOS_token = 1
    EOS_token = 2

    def __init__(self, hidden_size, embed_size, output_size, num_layers=1, vocab=None, attention=True):
        super(DecoderLSTM, self).__init__()
        if vocab is None:
            self.embedding = nn.Embedding(output_size, embed_size)

        self.dropout = nn.Dropout(0.5)
        if attention:
            self.LSTM = nn.LSTMCell(embed_size + 1280, hidden_size)
        else:
            self.LSTM = nn.LSTMCell(embed_size, hidden_size)

        # self.LSTM = nn.LSTM(embed_size + 2048, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)  # 2 for direction and 2 for image/hidden concat
        self.num_layers = num_layers
        self.attention = BahdanauAttention(hidden_size)
        if attention:
            self.attention_function = self.forward_step_bahdanau
        else:
            self.attention_function = self.forward_step
        self.init_h = nn.Linear(1280, hidden_size)  # linear layer to find initial hidden state of LSTM
        self.init_c = nn.Linear(1280, hidden_size)  # linear layer to find initial cell state of LSTM

        self.s_gate = nn.Linear(hidden_size, 1280)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        ####################################################
        # init weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, feature_outputs, caption, target_tensor=None, max_caption=80):
        caption_outputs, hidden, attention = self.generate_caption(feature_outputs, caption, target_tensor,
                                                                   max_length=max_caption)
        return caption_outputs, hidden, attention

    def generate_caption(self, feature_outputs, caption, target_tensor=None, max_length=80):
        decoder_outputs = []
        attentions = []
        batch_size = caption.size(0)

        max_length = max(max_length)

        target_embed = self.embedding(caption)
        decoder_input_start = torch.empty(batch_size, dtype=torch.long, device=device).fill_(self.SOS_token)
        decoder_input = self.embedding(decoder_input_start)
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
                if probability < 0.8: # 80-20 split between teacher forcing and greedy probability
                    decoder_input = target_embed[:, i,:]  # Teacher forcing
                else:
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(-1).detach()
                    decoder_input = self.embedding(decoder_input)

            else:  # validation/no teacher forcing
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
                decoder_input = self.embedding(decoder_input)

        decoder_outputs = torch.cat([tensor.unsqueeze(1) for tensor in decoder_outputs], dim=1)
        if attentions[0] is not None:
            attentions = torch.cat([tensor.unsqueeze(1) for tensor in attentions], dim=1)
        else:
            attentions = None
        return decoder_outputs, decoder_hidden, attentions  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden, image, image_feature):
        if type(hidden) is tuple:
            hidden, cell = hidden
        if image_feature: # Initial input is an image
            mean_encoder_out = image.mean(dim=1)
            hidden = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
            cell = self.init_c(mean_encoder_out)
        hidden, cell = self.LSTM(input, (hidden, cell))
        output = self.out(self.dropout(hidden))
        return output, (hidden, cell), output

    def forward_step_bahdanau(self, input, hidden, image, image_feature):
        if type(hidden) is tuple:
            hidden, cell = hidden
        if image_feature: # Initial input is an image
            mean_encoder_out = image.mean(dim=1)  # (batch_size, 196)
            hidden = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
            cell = self.init_c(mean_encoder_out)

        context, attn_weights = self.attention(hidden, image)
        gate = self.sigmoid(self.s_gate(hidden))
        context = context * gate
        input_lstm = torch.cat((input, context), dim=1)
        hidden, cell = self.LSTM(input_lstm, (hidden, cell))
        output = self.out(self.dropout(hidden))
        return output, (hidden,cell), attn_weights

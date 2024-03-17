import torch
import torch.nn as nn
from attention_models import BahdanauAttention, LuongAttn, SelfAttention
import torch.nn.functional as F

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class AttnDecoderGRU(nn.Module):
    MAX_LENGTH = 10
    SOS_token = 0
    EOS_token = 1

    def __init__(self, hidden_size, output_size, dropout_p=0.1, attention_type='bahdanau'):
        super(AttnDecoderGRU, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_size)
            self.attention_function = self.forward_step_bahdanau
            self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)

        elif attention_type == 'luong':
            self.attention = LuongAttn('general', hidden_size)
            self.attention_function = self.forward_step_luong
            self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.concat = nn.Linear(hidden_size * 2, hidden_size)


        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.MAX_LENGTH):

            decoder_output, decoder_hidden, attn_weights = self.attention_function(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        if attentions[0] is not None:
            attentions = torch.cat(attentions, dim=1)
        else:
            attentions = None

        return decoder_outputs, decoder_hidden, attentions

    def forward_step_bahdanau(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

    def forward_step_luong(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        attn_weights,_ = self.attention(output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)

        rnn_output = output.squeeze(1)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output.unsqueeze(1), hidden, None


class DecoderRNN(nn.Module):
    MAX_LENGTH = 10
    SOS_token = 0
    EOS_token = 1

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        # print("Encoder output")
        # print(encoder_outputs.shape)
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.SOS_token)
        # print("Decoder input")
        # print(decoder_input.shape)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input

                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
                # print("Teacher")
                # print(decoder_input.shape)

            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        print("Embedding")
        print(output.shape)
        print(hidden.shape)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # print("GRU OUTPUT")
        # print(output.shape)
        output = self.out(output)
        # print("DECODER OUTPUT")
        # print(output.shape)
        return output, hidden


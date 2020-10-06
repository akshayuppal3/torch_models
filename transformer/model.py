import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNetwork(nn.Module):
    def __init__(self, vocab_size, dim_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.pe = PositonalEncoding(dim_model, dropout)
        self.attn = MultiHeadAttention(dim_model, head_num=4)
        self.batchnorm = nn.LayerNorm(dim_model)
        self.l1 = nn.Linear(dim_model, d_ff)
        self.l2 = nn.Linear(d_ff, 2)
        self.droput = nn.Dropout(dropout)

    def forward(self, inputs):
        rnn_outputs = self.pe(self.embedding(inputs))
        outputs = rnn_outputs + self.attn(rnn_outputs, rnn_outputs, rnn_outputs) # deep residual connection
        outputs = self.batchnorm(outputs)
        return self.l2(self.droput(F.relu(self.l1(outputs))))


class PositonalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0., max_len).unsqueeze(1)  # position of each word
        div_term = torch.exp(torch.arange(0., dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('in_features({}) should be divisible by head_num({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.w_q = nn.Linear(in_features, in_features, bias)
        self.w_k = nn.Linear(in_features, in_features, bias)
        self.w_v = nn.Linear(in_features, in_features, bias)
        self.w_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)

        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.w_o(y)
        if self.activation is not None:
            y = self.activation(y)

        return y

    @staticmethod
    def gen_history_mask(x):
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self,x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
               .permute(0, 2, 1, 3) \
               .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)

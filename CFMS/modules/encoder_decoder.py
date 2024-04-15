from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from .utils import my_con_loss

from .att_model import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2),
                                            value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3),
                                         value.size(-1))

    selected_value = torch.gather(dummy_value, 3, dummy_idx)

    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, cmn, rm):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.cmn = cmn

        self.fuse_feature = nn.Linear(512 * 2, 512)
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix, cmn_masks=None, labels=None):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix,
                           cmn_masks=cmn_masks, labels=labels)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, memory_matrix=None, cmn_masks=None, labels=None):
        embeddings = self.tgt_embed(tgt)
        cmn_masks = cmn_masks.unsqueeze(1).expand(cmn_masks.shape[0], embeddings.size(1), cmn_masks.shape[-1])

        responses = self.cmn(embeddings, memory_matrix, memory_matrix, cmn_masks)
        embeddings = self.fuse_feature(torch.cat((embeddings, responses), dim=2))

        memory1 = self.rm.init_memory(memory.size(0)).to(memory)
        memory1 = self.rm(embeddings, memory1)

        return self.decoder(embeddings, memory, src_mask, tgt_mask, memory1), embeddings, responses


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask, memory1):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, memory1)
            return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, memory, src_mask, tgt_mask, memory1):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory1)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory1)
        return self.sublayer[2](x, self.feed_forward, memory1)


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory1):
        return x + self.dropout(sublayer(self.norm(x, memory1)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(rm_d_model, rm_d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory1):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        delta_gamma = self.mlp_gamma(memory1)
        delta_beta = self.mlp_beta(memory1)
        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout,
                                                  topk=self.topk)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.temp = vocab

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        memory1 = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory1 = torch.cat([memory1, pad], -1)
        elif self.d_model < self.num_slots:
            memory1 = memory1[:, :, :self.d_model]

        return memory1

    def forward_step(self, input, memory1):
        memory1 = memory1.reshape(-1, self.num_slots, self.d_model)
        q = memory1
        k = torch.cat([memory1, input.unsqueeze(1)], 1)
        v = torch.cat([memory1, input.unsqueeze(1)], 1)
        next_memory = memory1 + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory1))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory1
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory1):
        outputs = []
        for i in range(inputs.shape[1]):
            memory1 = self.forward_step(inputs[:, i], memory1)
            outputs.append(memory1)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab, cmn):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots,
                                 self.rm_d_model), self.num_layers),
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)), cmn, rm)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer, mode='train'):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.topk = args.topk
        self.num_cluster = args.num_cluster
        self.img_margin = args.img_con_margin
        self.txt_margin = args.txt_con_margin
        self.num_protype = args.num_protype

        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model

        self.img_cls_head = nn.Sequential(nn.Linear(args.cmm_dim, args.cmm_dim), nn.Linear(args.cmm_dim, 14))

        self.txt_cls_head = nn.Sequential(nn.Linear(args.cmm_dim, args.cmm_dim), nn.Linear(args.cmm_dim, 14))

        self.txt_dim_reduction = nn.Linear(args.d_txt_ebd, args.cmm_dim)
        self.dim_reduction = nn.Linear(args.d_txt_ebd + args.d_img_ebd, args.cmm_dim)

        self.img_dim_reduction = nn.Linear(args.d_img_ebd, args.cmm_dim)

        self.fuse_feature = nn.Linear(args.d_model * 2, args.d_model)
        tgt_vocab = self.vocab_size + 1
        self.cmn = MultiThreadMemory(args.num_heads, args.d_model, topk=args.topk)
        self.model = self.make_model(tgt_vocab, self.cmn)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        self.protypes = nn.Parameter(
            torch.FloatTensor(args.num_protype * args.num_cluster, args.d_txt_ebd + args.d_img_ebd))
        if mode == 'train':
            init_protypes = torch.load(args.init_protypes_path).float()
            self.protypes = nn.Parameter(init_protypes)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks, labels=None):
        att_feats, seq, att_masks, seq_mask, query_matrix, cmn_masks, _ = \
            self._prepare_feature_forward(att_feats, att_masks, labels=labels)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks, labels, query_matrix, cmn_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None, labels=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)

        max_num_protype = max(labels.sum(-1)) * self.num_protype
        protypes = self.dim_reduction(self.protypes)
        query_matrix = protypes.new_zeros(att_feats.size(0), max_num_protype.int(),
                                          protypes.shape[-1])
        cmn_masks = protypes.new_zeros(query_matrix.shape[0], att_feats.size(1), max_num_protype.int())

        for i in range(att_feats.size(0)):
            cur_query_matrix = []
            for j in range(len(labels[i])):
                if labels[i, j] == 1:
                    cur_query_matrix.extend(
                        protypes[j * self.num_protype:(j + 1) * self.num_protype, :])
            cur_query_matrix = torch.stack(cur_query_matrix, 0)

            query_matrix[i, :cur_query_matrix.shape[0], :] = cur_query_matrix
            cmn_masks[i, :, :cur_query_matrix.shape[0]] = 1

        responses = self.cmn(att_feats, query_matrix, query_matrix, cmn_masks)

        att_feats = self.fuse_feature(torch.cat((att_feats, responses), dim=2))

        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask, query_matrix, cmn_masks[:, 0, :], responses

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, labels=None):
        att_feats, seq, att_masks, seq_mask, query_matrix, cmn_masks, img_responses = \
            self._prepare_feature_forward(att_feats, att_masks, seq, labels)
        out, txt_feats, txt_responses = self.model(att_feats, seq, att_masks, seq_mask, memory_matrix=query_matrix,
                                                   cmn_masks=cmn_masks, labels=labels)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        img_con_loss = my_con_loss(torch.mean(img_responses, dim=1), num_classes=self.num_cluster,
                                   num_protypes=self.num_protype, labels=labels, margin=self.img_margin)
        img_con_loss = img_con_loss.unsqueeze(0)  # for  multi-gpu setting
        txt_con_loss = my_con_loss(torch.mean(txt_responses, dim=1), num_classes=self.num_cluster,
                                   num_protypes=self.num_protype, labels=labels, margin=self.txt_margin)
        txt_con_loss = txt_con_loss.unsqueeze(0)  # for  multi-gpu setting

        img_bce_loss = self.img_cls_head(torch.mean(att_feats, dim=1))
        txt_bce_loss = self.txt_cls_head(torch.mean(txt_feats, dim=1))

        return outputs, img_con_loss, txt_con_loss, img_bce_loss, txt_bce_loss

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask, query_matrix, cmn_masks, labels=None):
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        out, _, _ = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device),
                                      memory_matrix=query_matrix, cmn_masks=cmn_masks, labels=labels)

        return out[:, -1], [ys.unsqueeze(0)]

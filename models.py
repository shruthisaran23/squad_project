"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        dropProbability (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, dropProbability=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    dropProbability=dropProbability)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     dropProbability=dropProbability)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         dropProbability=dropProbability)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     dropProbability=dropProbability)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      dropProbability=dropProbability)

    def forward(self, cweightIndex, qweightIndex):
        charMask = torch.zeros_like(cweightIndex) != cweightIndex
        quesMask = torch.zeros_like(qweightIndex) != qweightIndex
        lenC, lenQ = charMask.sum(-1), quesMask.sum(-1)

        c_emb = self.emb(cweightIndex)         # (batch_size, lenC, hidden_size)
        q_emb = self.emb(qweightIndex)         # (batch_size, lenQ, hidden_size)

        c_enc = self.enc(c_emb, lenC)    # (batch_size, lenC, 2 * hidden_size)
        q_enc = self.enc(q_emb, lenQ)    # (batch_size, lenQ, 2 * hidden_size)
        att = self.att(c_enc, q_enc,
                       charMask, quesMask)    # (batch_size, lenC, 8 * hidden_size)
        mod = self.mod(att, lenC)        # (batch_size, lenC, 2 * hidden_size)
        out = self.out(att, mod, charMask)  # 2 tensors, each (batch_size, lenC)
        return out

class SelfAttentionPlusCharacter(nn.Module):
    def __init__(self, wrdVectors, numberChars, embeddingSize, maximumWordLength, hidden_size, dropProbability=0.):
        super(SelfAttentionPlusCharacter, self).__init__()
        self.hidden_size = hidden_size
        self.characterEmbed = layers.CharacterEmbed(numberChars=n_chars, embeddingSize=embed_size, maximumWordLength=maximumWordLength,
                                    hidden_size=hidden_size,
                                    dropProbability=dropProbability)
        self.Embedding = layers.Embedding(wrdVectors=wrdVectors,
                                    hidden_size=hidden_size,
                                    dropProbability=dropProbability)
        self.attentionIntial = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         dropProbability=dropProbability)
        self.encoder = layers.RNNEncoder(input_size=2*hidden_size,
                                     hidden_size=hidden_size,
                                     nLayers=1,
                                     dropProbability=dropProbability)
        self.selfAttention = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                 dropProbability=dropProbability)
        self.output = layers.BiDAFOutput(hidden_size=hidden_size,
                                      dropProbability=dropProbability, multiplier=14)

    def forward(self, charIndex, queIndex, cweightIndex, qweightIndex):
        charMask = torch.zeros_like(cweightIndex) != cweightIndex
        #print("charMask", charMask)
        quesMask = torch.zeros_like(qweightIndex) != qweightIndex
        #print("quesMask", quesMask)
        lenC, lenQ = charMask.sum(-1), quesMask.sum(-1)
        #concatenate 
        c_emb = torch.cat([self.charEmb(charIndex) , self.wordEmb(cweightIndex)  ], dim=-1)
        q_emb = torch.cat([self.charEmb(queIndex)  , self.wordEmb(qweightIndex) ], dim=-1)
        attn1 = self.enc(c_emb, lenC) 
        #print("1", attn1)
        attn2 = self.enc(q_emb, lenQ),charMask, quesMask
        #print("2", attn2)
        attentionIntial = self.att(attn1, attn2)   
         
        charAttention = attentionIntial[:, :, 2*self.hidden_size:4*self.hidden_size] 
        self_att = self.self_att(charAttention, charAttention, charMask, charMask)
        attentionIntial = torch.cat([att, self_att[:, :, 2*self.hidden_size:]], dim=-1)      
        out = self.out(att, self.mod(att, lenC), charMask)  

        return out
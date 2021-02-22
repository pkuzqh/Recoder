import torch.nn as nn

from Multihead_Attention import MultiHeadedAttention
from SubLayerConnection import SublayerConnection
from DenseLayer import DenseLayer
from ConvolutionForward import ConvolutionLayer
from Multihead_Combination import MultiHeadedCombination


class decodeTransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention1 = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.attention2 = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = DenseLayer(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.sublayer1 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer3 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, inputleft, leftmask, inputleft2, leftmask2):
        x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, inputleft, inputleft, mask=leftmask))
        x = self.sublayer3(x, lambda _x: self.attention2.forward(_x, inputleft2, inputleft2, mask=leftmask2))
        x = self.sublayer4(x, self.feed_forward)
        return self.dropout(x)
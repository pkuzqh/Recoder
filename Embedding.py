import torch.nn as nn
from TokenEmbedding import TokenEmbedding
from postionEmbedding import PositionalEmbedding


class Embedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.depth_embedding = nn.Embedding(20, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, inputdept=None, usedepth=False):
        x = self.token(sequence) + self.position(sequence)
        if usedepth:
            x = x + self.depth_embedding(inputdept)
        return self.dropout(x)
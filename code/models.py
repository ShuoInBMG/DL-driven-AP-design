import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        
        # Compute dot product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim), keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        
        # Concatenate heads output
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        forward = self.dropout(forward)
        out = self.norm2(forward + x)
        return out

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, output_size, embed_size=32, num_layers=6, heads=8, dropout=0, forward_expansion=4):
        super(TimeSeriesTransformer, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embed_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 全局平均池化层
        self.fc_out = nn.Linear(embed_size, output_size)
    
    def forward(self, x, mask):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        x = x.permute(0, 2, 1)  # 将张量维度调整为(batchsize, embed_size, seq_len)
        x = self.pooling(x)  # 全局平均池化，将seq_len维度压缩为1
        x = x.squeeze(2)  # 去除维度为1的seq_len维度
        out = self.fc_out(x)
        return out.squeeze()  # 将维度从(batchsize, 1)降低到(batchsize,)
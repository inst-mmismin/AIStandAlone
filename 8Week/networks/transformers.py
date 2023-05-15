import torch 
import math 
import torch.nn as nn 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model # 임배딩의 출력 값 
        self.num_heads = num_heads 
        self.head_dim = d_model // num_heads # 하나의 해드에 할당될 단어 차원의 크기 

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v): # q,k,v : [batch_size, seq_len, d_model] 
        b, seq_len, _ = q.shape

        Q = self.q(q)
        K = self.k(k)
        V = self.v(v)

        # Q, K, V 크기 변환 : [batch_size, seq_len, d_model] ->  [batch_size, seq_len, num_heads, head_dim]
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim)
        K = K.view(b, seq_len, self.num_heads, self.head_dim)
        V = V.view(b, seq_len, self.num_heads, self.head_dim)

        # attention score 계산
        # [batch_size, seq_len, num_heads, head_dim] * [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, num_heads, num_heads]
        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) 

        # attention score를 softmax를 통해 확률값으로 변환
        attention_weights = torch.softmax(score, dim=-1) # [batch_size, seq_len, num_heads, num_heads]

        # attention score를 value(V)에 곱해 최종 결과값을 얻음
        attention = torch.matmul(attention_weights, V) # [batch_size, seq_len, num_heads, head_dim]

        # attention head 마다 계산된 결과를 하나로 연결 (concatenate)
        attention = attention.reshape(b, seq_len, self.d_model)

        # 최종 출력
        out = self.out(attention) 
        
        return out 


class TransformersLayer(nn.Module):
    def __init__(self, d_model, num_heads, feedforward_dim=2048, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward  = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # self attention 
        skip = x.clone()
        x = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = self.norm1(x + skip)

        # feedforward
        skip = x.clone()
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.norm2(x + skip)
        return x 

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, max_seq): 
        super().__init__()
        positional_encoding = torch.zeros(max_seq, d_model)

        pos = torch.arange(0, max_seq, dtype=torch.float).unsqueeze(1)
        dim = torch.arange(0, d_model, step=2, dtype=torch.float)
        div_term = torch.exp(-math.log(10000) * (dim // d_model))

        positional_encoding[:, 0::2] = torch.sin(pos * div_term)
        positional_encoding[:, 1::2] = torch.cos(pos * div_term) 

        # 학습 변수가 아닌 상수로 저장하며
        # 모델이 학습되는 동안 학습되지 않도록 함
        # 그러면서도 모델의 일부로 포함되어 GPU 연산등이 가능함 
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0)) # 마지막 unsqueeze는 batch 차원 추가

    def forward(self, x): 
        x = x + self.positional_encoding[:, :x.size(1)] 
        return x 

class TransformersEncoder(nn.Module):
    def __init__(self, vocab_size, max_length, num_layer, d_model, num_heads, feedforward_dim=2048, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq=max_length)
        layers = []
        for _ in range(num_layer): 
            layers.append(TransformersLayer(d_model, num_heads, feedforward_dim, dropout_rate))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x) 
        x = self.positional_encoding(x)
        x = self.layers(x)
        return x 


class IMDBClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = TransformersEncoder(args.vocab_size, args.max_length, args.num_layer, args.d_model, args.num_heads, args.feedforward_dim, args.dropout_rate)
        self.classifier = nn.Linear(args.d_model, args.num_classes)

    def forward(self, x):
        x = self.encoder(x) # [batch_size, seq_len, d_model]
        x = x[:, 0, :] # 가장 첫 위치에 있는 토큰의 임배딩만 사용 
        x = self.classifier(x) # [batch_size, num_classes]
        return x 

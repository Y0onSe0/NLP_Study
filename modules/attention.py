import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # key, value, query에 대한 선형변환 layer 초기화.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # 이 드롭아웃은 트랜스포머의 원래 구현에 따라 normalized attention scores에 적용된다.
    # 다소 이례적이지만, 경험적으로 이것이 더 나은 성능을 제공한다고 알려져 있다.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # hidden_state (x) 를 사영하기 위해 k, v, q의 해당 linear_layer가 사용된다.
    proj = linear_layer(x)
    # 다음으로, 프로젝션에 대해 여러 헤드를 생성해야 한다. 
    # 이는 은닉 상태를 self.num_attention_heads로 분할하며, 
    # 각 헤드는 self.attention_head_size 크기를 갖도록 한다.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # 적절히 전치하여 크기 [bs, num_attention_heads, seq_len, attention_head_size]인 프로젝션을 얻는다.
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    # key, query, value: [bs, num_heads, seq_len, head_size]
    d_k = self.attention_head_size
    seq_len = query.size(-2)

    # Scaled dot-product attention scores: [bs, num_heads, seq_len, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    # Causal mask: 미래 토큰을 보지 못하도록 상삼각 부분을 -10000으로 마스킹
    # shape: [1, 1, seq_len, seq_len]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device, dtype=query.dtype))
    causal_mask = (1.0 - causal_mask) * -10000.0
    scores = scores + causal_mask

    # attention_mask: [bs, 1, 1, seq_len] — padding 위치는 큰 음수 → softmax 후 ≈ 0
    scores = scores + attention_mask

    # Softmax로 attention weight 계산
    attn_weights = torch.softmax(scores, dim=-1)

    # 원래 Transformer 구현 방식에 따라 attention weight에 dropout 적용
    attn_weights = self.dropout(attn_weights)

    # Value와 가중합: [bs, num_heads, seq_len, head_size]
    context = torch.matmul(attn_weights, value)

    # 멀티헤드 병합: [bs, seq_len, all_head_size]
    context = rearrange(context, 'b h t d -> b t (h d)')

    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # 먼저, self.transform을 사용하여 multi-head attention에 필요한
    # 각 토큰의 key, value, query를 생성해야 한다(함수 내부에 자세한 내용 있음).
    # *_layer의 크기 = [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # multi-head attention 계산.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value

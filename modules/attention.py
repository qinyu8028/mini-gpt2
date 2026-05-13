import torch

from einops import rearrange
from torch import nn, softmax


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original implementation of transformer.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    self.use_lora = config.use_lora

    if self.use_lora == True:
      self.lora_query = LoRA(config)
      self.lora_value = LoRA(config)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # produce multiple heads for the proj. 
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    proj = rearrange(proj, 'b t h d -> b h t d')
    # size [bs, num_attention_heads, seq_len, attention_head_size]
    return proj

  def attention(self, key, query, value, attention_mask):
    dk = key.shape[-1]
    T = key.shape[-2]
    score = query @ key.transpose(-2, -1) / (dk ** 0.5)
    tri = torch.triu(torch.ones(T, T, device=query.device), diagonal=1).bool()
    masked_score = score.masked_fill(tri, float('-inf'))
    masked_score = masked_score + attention_mask
    weight = softmax(masked_score, dim=-1)
    weight_dropped = self.dropout(weight)
    attn_value = weight_dropped @ value
    attn_value = rearrange(attn_value, 'b h t d -> b t (h d)')
    
    return attn_value

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    if not self.use_lora:
      # First, we have to generate the key, value, query for each token for multi-head attention
      # using self.transform (more details inside the function).
      # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
      key_layer = self.transform(hidden_states, self.key)
      value_layer = self.transform(hidden_states, self.value)
      query_layer = self.transform(hidden_states, self.query)
      
      # Calculate the multi-head attention.
      attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)

    else:
      key_layer = self.transform(hidden_states, self.key)
      
      value_lora = self.lora_value(hidden_states, self.value)
      query_lora = self.lora_query(hidden_states, self.query)

      value_lora_layer = rearrange(value_lora, 'b t (h d) -> b h t d', h=self.num_attention_heads)
      query_lora_layer = rearrange(query_lora, 'b t (h d) -> b h t d', h=self.num_attention_heads)
      # size [bs, num_attention_heads, seq_len, attention_head_size]
      
      attn_value = self.attention(key_layer, query_lora_layer, value_lora_layer, attention_mask)

    return attn_value


class LoRA(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.hidden_state = config.hidden_size
    self.r = config.r
    self.alpha = config.alpha
    
    self.W_a = nn.Linear(self.hidden_state, self.r)  
    self.W_b = nn.Linear(self.r, self.hidden_state)

    self.param_initialize()

  def param_initialize(self):
    nn.init.zeros_(self.W_b.weight)
    nn.init.zeros_(self.W_b.bias)

    nn.init.normal_(self.W_a.weight)
    nn.init.zeros_(self.W_a.bias)

  def forward(self, input_x, W_0: nn.Module):
    out_a = self.W_a(input_x)
    out_b = self.W_b(out_a)
    out = W_0(input_x) + (self.alpha / self.r) * out_b

    return out
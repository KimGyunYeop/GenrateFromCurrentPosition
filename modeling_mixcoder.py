from torch import nn
import torch

class MixCoderSelfAttention(nn.Module):
    def __init__(self, num_head=8, dim_head=64, layer_type="encoder") -> None:
        super().__init__()

        self.num_head = num_head
        self.dim_head = dim_head

        self.scaling = self.dim_head**-0.5

        self.q_proj = nn.Linear(self.num_head*self.dim_head, self.num_head*self.dim_head)
        self.k_proj = nn.Linear(self.num_head*self.dim_head, self.num_head*self.dim_head)
        self.v_proj = nn.Linear(self.num_head*self.dim_head, self.num_head*self.dim_head)
        self.out_proj = nn.Linear(self.num_head*self.dim_head, self.num_head*self.dim_head)

        self.layer_type = layer_type

    def forward(self, hidden_states, attention_mask=None):

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(query_states.size(0), query_states.size(1), self.num_head, self.dim_head)
        key_states = key_states.view(key_states.size(0), key_states.size(1), self.num_head, self.dim_head)
        value_states = value_states.view(value_states.size(0), value_states.size(1), self.num_head, self.dim_head)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_scores = attn_scores / self.scaling

        attn_scores = attn_scores + attention_mask
        attn_scores = nn.Softmax(dim=-1)(attn_scores)

        attn_output = torch.matmul(attn_scores, value_states)

        return attn_output, attn_scores


class MixCoderEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        self.self_attn = MixCoderSelfAttention(config.num_head, config.dim_head)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        

        pass

class MixCoderDecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        self.self_attn = MixCoderSelfAttention(config.num_head, config.dim_head, layer_type="decoder")
        self.encoder_attn = MixCoderSelfAttention(config.num_head, config.dim_head, layer_type="encoder")
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        pass

class MixCoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if not config.only_decoder:
            self.encoder = nn.ModuleList([MixCoderEncoderLayer(config) for _ in range(config.num_layer)])
        
        self.decoder = nn.ModuleList([MixCoderDecoderLayer(config) for _ in range(config.num_layer)])
    
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        
        hidden_states = self.embeddings(input_ids)

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
    
        for enc_layer in self.encoder:
            hidden_states = enc_layer(hidden_states, attention_mask)

        decoder_attention_mask = decoder_attention_mask.unsqueeze(1).unsqueeze(2)
        decoder_attention_mask = (1.0 - decoder_attention_mask) * -10000.0

        decoder_input_ids = self.embeddings(decoder_input_ids)

        for dec_layer in self.decoder:
            hidden_states = dec_layer(hidden_states, decoder_attention_mask)
        
        pass

class MixCoderForConditionalGeneration(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        self.model = MixCoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.weight_tie()

    def weight_tie(self):
        self.lm_head.weight = self.model.embeddings.weight

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels=None):

        out = self.model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
        
        print(out)
        
        pass

class MixCoderConfig():
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_head = kwargs.get("num_head", 8)
        self.dim_head = kwargs.get("dim_head", 64)
        self.num_layer = kwargs.get("num_layer", 6)
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.only_decoder = kwargs.get("only_decoder", False)
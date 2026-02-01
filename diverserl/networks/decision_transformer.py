from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diverserl.common.type_aliases import _activation, _initializer, _kwargs
from diverserl.networks.utils import get_activation, get_initializer


class DecisionTransformer(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 128,
                 episode_len: int = 1000,
                 sequence_length: int = 10,
                 layer_num: int = 3,
                 heads_num: int = 1,
                 dropout: float = 0.1,
                 activation: Optional[_activation] = nn.ReLU,
                 activation_kwargs: Optional[_kwargs] = None,
                 embedding_weight_initializer: Optional[_initializer] = nn.init.normal_,
                 embedding_weight_initializer_kwargs: Optional[_kwargs] = None,
                 embedding_bias_initializer: Optional[_initializer] = nn.init.zeros_,
                 embedding_bias_initializer_kwargs: Optional[_kwargs] = None,
                 layer_norm_weight_initializer: Optional[_initializer] = nn.init.ones_,
                 layer_norm_weight_initializer_kwargs: Optional[_kwargs] = None,
                 layer_norm_bias_initializer: Optional[_initializer] = nn.init.zeros_,
                 layer_norm_bias_initializer_kwargs: Optional[_kwargs] = None,
                 device: str = "cpu",
                 ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.episode_len = episode_len
        self.sequence_length = sequence_length
        self.layer_num = layer_num
        self.heads_num = heads_num
        self.dropout = dropout

        self.device = device

        self.activation, self.activation_kwargs = get_activation(activation, activation_kwargs)
        self.embedding_weight_initializer, self.embedding_weight_initializer_kwargs = get_initializer(
            embedding_weight_initializer,
            embedding_weight_initializer_kwargs)
        self.embedding_bias_initializer, self.embedding_bias_initializer_kwargs = get_initializer(
            embedding_bias_initializer,
            embedding_bias_initializer_kwargs)
        self.layer_norm_weight_initializer, self.layer_norm_weight_initializer_kwargs = get_initializer(
            layer_norm_weight_initializer,
            layer_norm_weight_initializer_kwargs)
        self.layer_norm_bias_initializer, self.layer_norm_bias_initializer_kwargs = get_initializer(
            layer_norm_bias_initializer,
            layer_norm_bias_initializer_kwargs)

        self.embedding_dropout = nn.Dropout(dropout)

        self.embedding_norm = nn.LayerNorm(embedding_dim)

        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)
        self.timestep_emb = nn.Embedding(episode_len + sequence_length, embedding_dim)

        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=heads_num, batch_first=True),
            num_layers=layer_num
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, action_dim),
            nn.Tanh()
        )

        self.to(torch.device(device))

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initialize layer weights and biases from wanted initializer specs.

        :param m: a single torch layer
        """
        if isinstance(m, (nn.Linear, nn.Embedding)):
            if self.embedding_weight_initializer is not None:
                self.embedding_weight_initializer(m.weight, **self.embedding_weight_initializer_kwargs)
            if m.bias is not None and self.embedding_bias_initializer is not None:
                self.embedding_bias_initializer(m.bias, **self.embedding_bias_initializer_kwargs)
        elif isinstance(m, nn.LayerNorm):
            if self.layer_norm_weight_initializer is not None:
                self.layer_norm_weight_initializer(m.weight,
                                                   **self.layer_norm_weight_initializer_kwargs)
            if m.bias is not None and self.layer_norm_bias_initializer is not None:
                self.layer_norm_bias_initializer(m.bias, **self.layer_norm_bias_initializer_kwargs)

    def forward(
            self,
            states: torch.Tensor,  # [batch_size, sequence_length, state_dim]
            actions: torch.Tensor,  # [batch_size, sequence_length, action_dim]
            returns: torch.Tensor,  # [batch_size, sequence_length]
            timesteps: torch.Tensor,  # [batch_size, sequence_length]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, sequence_length]
    ) -> torch.FloatTensor:
        batch_size, sequence_length = states.shape[0], states.shape[1]

        t_emb = self.timestep_emb(timesteps)
        s_emb = self.state_emb(states) + t_emb
        a_emb = self.action_emb(actions) + t_emb
        returns_emb = self.return_emb(returns) + t_emb

        sequence = (
            torch.stack([returns_emb, s_emb, a_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * sequence_length, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, sequence_length * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * sequence_length)
            )
        out = self.embedding_norm(sequence)
        out = self.embedding_dropout(out)

        out = self.blocks(out, src_key_padding_mask=padding_mask)
        out = self.head(out)

        predicted_action = out[:, 1::3]

        return predicted_action

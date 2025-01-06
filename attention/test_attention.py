import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from build.llm_nodes import layers

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    if cos.dim() == 2:
        cos = cos[: x.size(-2), :]
        sin = sin[: x.size(-2), :]
    elif cos.dim() == 3:
        cos = cos[:, : x.size(-2), :]
        sin = sin[:, : x.size(-2), :]
    elif cos.dim() == 4:
        cos = cos[:, :, : x.size(-2), :]
        sin = sin[:, :, : x.size(-2), :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbeddingESM(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """
    def __init__(self, dim: int, base: Union[int, float] = 10000,
                 distance_scale: Union[int, float] = 1, dtype=torch.half):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale
        self.dtype = dtype

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq.to(dtype))

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

        self.apply_rotary_pos_emb = apply_rotary_pos_emb

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.size(seq_dimension)
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            if x.dim() == 2:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
            elif x.dim() == 3:
                self._cos_cached = emb.cos()[None, :, :]
                self._sin_cached = emb.sin()[None, :, :]
            elif x.dim() == 4:
                self._cos_cached = emb.cos()[None, None, :, :]
                self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
                self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached))


class Linear(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dtype: torch.dtype = torch.half,
                 init_mean: float = 0.0, init_std: float = 1, scale_before: bool = True):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale_before = scale_before
        self.weight = torch.nn.parameter.Parameter(torch.empty((dim_out, dim_in), dtype=dtype))
        torch.nn.init.normal_(self.weight, mean=init_mean, std=init_std)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale_before:
            x = x / math.sqrt(self.dim_in)
            x = F.linear(x, self.weight)
        else:
            x = F.linear(x, self.weight)
            x = x / math.sqrt(self.dim_in)
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dim_head: int, 
                 pos_bias_type: Optional[str] = "rotary", 
                 dtype: torch.dtype = torch.half,
                 dropout_p: Optional[float] = None) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.pos_bias_type = pos_bias_type

        self.project_q = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_k = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_v = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)

        self.attn_out = Linear(self.num_heads * self.dim_head, self.dim_model, dtype=dtype)
        self.position_bias = RotaryEmbeddingESM(dim = dim_head, dtype=dtype)
        self.softmax = torch.nn.Softmax(dim=-1)

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=dropout_p)
        else:
            self.dropout = None


    def forward(
        self,
        hidden_q: torch.Tensor,
        attention_mask: torch.BoolTensor,
        position_bias: torch.Tensor,
    ):
        """
        Args:
            hidden_q (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.
        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """  # noqa: E501
        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(1)

        h_q = self.project_q(hidden_q)
        h_k = self.project_k(hidden_q)
        h_v = self.project_v(hidden_q)

        h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        h_k = h_k.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        h_v = h_v.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        if self.pos_bias_type == "rotary":
            # b h s d
            h_q, h_k = self.position_bias(h_q, h_k)

        # res = h_v.permute(0, 2, 1, 3)
        # print("h_v: ", res.shape, res)
        # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
        score = torch.matmul(h_q, h_k.transpose(-1, -2)) / math.sqrt(
            self.dim_head
        )  # cpm-live does scaling before rotary embedding.

        score = torch.masked_fill(score,
            attention_mask.view(batch_size, 1, len_q, len_q) == False,
            torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),)

        score = self.softmax(score)

        # print(score)

        score = torch.masked_fill(score,
            attention_mask.view(batch_size, 1, len_q, len_q) == False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype),)

        if self.dropout is not None:
            score = self.dropout(score)

        # (b, n_h, len_q, len_k) @ (b, n_h, len_k, d_h) -> (b, n_h, len_q, d_h)
        score = torch.matmul(score, h_v)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(
            0, 2, 1, 3
        )
        score = score.contiguous().view(
            batch_size, len_q, self.num_heads * self.dim_head
        )
        return self.attn_out(score)


def test_attention(batch, shapes, seqlen, trans, flash_decoding):
    rtol, atol = (1e-3, 2e-2)
    print(shapes)
    num_heads, dim_head = shapes
    dim_model = num_heads * dim_head
    hidden = torch.randn([batch, seqlen, dim_model], dtype=torch.half, device="cuda")
    position_bias = (torch.arange(seqlen, dtype=torch.int32, device="cuda",).repeat((batch, 1)))
    mask = ((torch.arange(seqlen, device="cuda") <= torch.arange(seqlen, device="cuda").view(-1, 1))
            .to(torch.int8).repeat(batch, 1).view(batch, seqlen, seqlen))
    print(mask.shape)
    seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"),
                           torch.cumsum(torch.amax(position_bias, dim=-1) + 1, dim=-1, dtype=torch.int32),])
    if flash_decoding:
        seqlens_kv = torch.zeros_like(torch.amax(position_bias, dim=-1))
    else:
        seqlens_kv = torch.empty([0], dtype=torch.int32)

    attn_pt = Attention(dim_model, num_heads, dim_head, "rotary").cuda(0)
    state_dict_pt = attn_pt.state_dict()

    out_pt = attn_pt.forward(hidden, mask, position_bias)
    print(out_pt.shape)
    print("===" * 20)


if __name__ == "__main__":
    test_attention(4, (4, 16), 17, False, False)
    # test_attention(4, (4, 128), 257, False, False)

import math
import torch
import torch.nn.functional as F
from typing import Optional, Union
from build.llm_nodes import layers

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, 
                 dim,
                 base=10000,
                 distance_scale: Union[int, float] = 1,
                 dtype: torch.dtype = torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim))
        inv_freq = inv_freq.to(dtype)
        self.distance_scale = distance_scale
        self.dtype = dtype
        self.inv_freq = inv_freq

    def forward(self, x: torch.Tensor, x_pos: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(..., dim)``): Inputs.
            x_pos (:obj:`torch.Tensor` of shape ``(...)``): Positions of inputs.
        """
        x_pos = x_pos * self.distance_scale
        freqs = x_pos[..., None].to(self.dtype) * self.inv_freq[None, :]  # (..., dim/2)

        # the same implementation as sat
        emb = torch.cat((freqs, freqs), dim=-1) # (..., dim)
        emb_cos = emb.cos()   # (..., dim)
        emb_sin = emb.sin()   # (..., dim)

        rotate_x = torch.cat(
            [-x[..., x.size(-1) // 2 :], x[..., : x.size(-1) // 2]], dim=-1)  # (..., dim)

        return x * emb_cos + rotate_x * emb_sin


class EmbeddingExt(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 dtype: torch.dtype = torch.half,
                 init_mean: float = 0.0,
                 init_std: float = 1,
                 distance_scale: int = 16,):
        super().__init__()

        self.dim_model = embedding_size
        self.rotary_emb = RotaryEmbedding(
            dim=embedding_size, distance_scale=distance_scale, dtype=dtype)
        self.weight = torch.nn.parameter.Parameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),)
        torch.nn.init.normal_(self.weight, mean=init_mean, std=init_std)

    def forward(self, ids: torch.Tensor, ids_sub: torch.Tensor):
        """
        Args:
            ids (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            ids (:obj:`torch.Tensor` of shape ``(batch_size)``): Subscript of input sequence tokens.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        embeds = F.embedding(ids, self.weight) / math.sqrt(self.dim_model)
        return self.rotary_emb(embeds, ids_sub)

    def projection(self, x: torch.Tensor, ext_table: Optional[torch.Tensor] = None):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
            ext_table (:obj:`torch.Tensor` of shape ``(ext_table_size, dim_model)``): Ext vocab table.
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_size + ext_table_size)``: The projection output.
        """  # noqa: E501
        logits = F.linear(x / math.sqrt(self.dim_model), self.weight)
        if ext_table is not None:
            logits_ext = F.linear(x, ext_table)
            logits = torch.cat([logits, logits_ext], dim=-1)
        return logits


def test_embedding(SIZE, BATCH, SEQLEN, SCALE):
    rtol, atol = (1e-3, 5e-3)

    input = torch.randint(0, SIZE[0], (BATCH, SEQLEN,), dtype=torch.int32, device="cuda")
    input_subs = torch.randint(0, SEQLEN, (BATCH, SEQLEN,), dtype=torch.int32, device="cuda")

    ff_pt = EmbeddingExt(SIZE[0], SIZE[1], dtype=torch.half).cuda()
    out_pt = ff_pt.forward(input, input_subs)
    print(out_pt.shape)

    ff = layers.Embedding(SIZE[1], SIZE[0], SCALE)
    state_dict_pt = ff_pt.state_dict(prefix="token_embedding.")
    ff.load_state_dict(
        dict([(k, v.contiguous().cpu().numpy()) for k, v in state_dict_pt.items()]))
    state_dict = ff.named_parameters()
    print(state_dict)

    for name, param in state_dict_pt.items():
        assert name in state_dict
        assert torch.allclose(
            torch.from_numpy(state_dict[name]).to(torch.half),
            param.cpu(),
            rtol=rtol,
            atol=atol)
    out = ff.forward(input.cpu().numpy(), input_subs.cpu().numpy())
    print(out)
    assert torch.allclose(
        torch.from_numpy(out).to(torch.half), out_pt.cpu(), rtol=rtol, atol=atol)
 
if __name__ == "__main__":
    test_embedding((2, 8), 4, 4, True)


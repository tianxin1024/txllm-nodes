import math
import torch
import torch.nn.functional as F

from build.llm_nodes import layers


class Linear(torch.nn.Module):
    def __init__(self, 
                 dim_in: int,
                 dim_out: int,
                 act_fn_type: str = "",
                 dtype: torch.dtype = torch.half,
                 init_mean: float = 0.0,
                 init_std: float = 1,
                 scale_before: bool = True):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.act_fn_type = act_fn_type
        self.weight = torch.nn.parameter.Parameter(
            torch.empty((dim_out, dim_in),dtype=dtype))
        self.scale_before = scale_before
        if act_fn_type.lower() == "gelu":
            self.act = torch.nn.GELU()
        elif act_fn_type.lower() == "silu":
            self.act = torch.nn.SiLU()
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

        if self.act_fn_type:
            x = self.act(x)
        return x
 

def test_linear(SIZE, BATCH, SEQLEN, ACTIVATION, SCALE, W_TRANS, DTYPE):
    if DTYPE == torch.bfloat16:
        rtol, atol = (1e-2, 3e-3)
    else:
        rtol, atol = (1e-3, 3e-3)

    input = torch.randn([BATCH, SEQLEN, SIZE[0]], dtype=DTYPE, device="cuda")
    input.requires_grad = True

    ff_pt = Linear(SIZE[0], SIZE[1], ACTIVATION, dtype=DTYPE, scale_before=SCALE).cuda()
    out_pt = ff_pt.forward(input)
    # print(out_pt)

    ff = layers.Linear(SIZE[0], SIZE[1], ACTIVATION, False, SCALE, W_TRANS, 
                       "bfloat" if DTYPE == torch.bfloat16 else "half")
    # print(ff)
    state_dict_pt = ff_pt.state_dict()
    weight_pt = state_dict_pt["weight"]
    print(state_dict_pt)

    state_dict_ff = dict([(k, v.contiguous().cpu().numpy()) for k, v in state_dict_pt.items()])
    print(state_dict_ff)
    ff.load_state_dict(state_dict_ff)

    state_dict = ff.named_parameters()
    print(state_dict)

    for name, param in state_dict_pt.items():
        assert name in state_dict
        assert torch.allclose(torch.from_numpy(state_dict[name]).to(torch.half), param.cpu(), rtol=rtol, atol=atol)



if __name__ == "__main__":
    test_linear((5, 6), 4, 2, "", True, False, torch.half)

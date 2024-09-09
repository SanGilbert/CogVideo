import torch
import torch.nn as nn
from packaging import version
from .util import cross_norm

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        for key in c:
            c[key] = c[key].to(self.dtype)

        # use concat key to add, simplify training flow.
        if "concat" in c.keys():
            add_c = c["concat"].permute(0, 2, 1, 3, 4) # B, C, T, H, W -> B, T, C, H, W
            assert add_c.shape[2] % x.shape[2] == 0
            num_c = add_c.shape[2] // x.shape[2]
            add_c = add_c.chunk(num_c, dim=2)
            add_c = sum(add_c)
            x = x + add_c
            # controlnext
            # x = cross_norm(x, add_c, scale=0.2)
            
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )

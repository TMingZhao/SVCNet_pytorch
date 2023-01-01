
from torch.autograd import Function
import torch
import sampling

class _TestFunction(Function):
    @staticmethod
    def forward(self, b, c, n, npoints, points_tensor, idx_tensor, out_tensor):
        """
        It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
        The context can be used to store tensors that can be then retrieved
        during the backward pass."""
        return sampling.gather_forward(b, c, n, npoints, points_tensor, idx_tensor, out_tensor)

    @staticmethod
    def backward(self, b, c, n, npoints, grad_out_tensor, idx_tensor, grad_points_tensor):
        return sampling.gather_backward(b, c, n, npoints, grad_out_tensor, idx_tensor, grad_points_tensor)

b = _TestFunction.apply



class GatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor
        idx : torch.Tensor
            (B, npoint) tensor of the features to gather
        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
        features = features.contiguous()
        idx = idx.contiguous()
        idx = idx.to(dtype=torch.int32)

        B, npoint = idx.size()
        _, C, N = features.size()

        output = torch.empty(
            B, C, npoint, dtype=features.dtype, device=features.device)
        output = sampling.gather_forward(
            B, C, N, npoint, features, idx, output
        )

        ctx.save_for_backward(idx)
        ctx.C = C
        ctx.N = N
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, = ctx.saved_tensors
        B, npoint = idx.size()

        grad_features = torch.zeros(
            B, ctx.C, ctx.N, dtype=grad_out.dtype, device=grad_out.device)
        grad_features = sampling.gather_backward(
            B, ctx.C, ctx.N, npoint, grad_out.contiguous(), idx, grad_features
        )

        return grad_features, None


gather_points = GatherFunction.apply

# # 封装成一个模块（Module）
# class Test(torch.nn.Module):
#     def __init__(self):
#         super(Test, self).__init__()

#     def forward(self, inputA, inputB):
#         return _TestFunction.apply(inputA, inputB)
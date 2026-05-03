import torch


class FullyConnected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        """
        Computes the output of the fully_connected function given in the assignment

        Arguments
        ---------
        ctx: a PyTorch context object
        x (Tensor): of size (T x n), the input features
        w (Tensor): of size (n x m), the weights
        b (Tensor): of size (m), the biases

        Returns
        -----
        y (Tensor): of size (T x m), the outputs of the fully_connected operator
        """
        # Save inputs needed to apply the chain rule in backward.
        ctx.save_for_backward(x, w, b)
        # Batched fully connected layer: each row of x is one example.
        y = x @ w + b

        return y

    @staticmethod
    def backward(ctx, dz_dy):
        """
        back-propagates the gradients with respect to the inputs
        ctx: a PyTorch context object.
        dz_dy (Tensor): of size (T x m), the gradients with respect to the output argument y

        Returns
        -------
        dzdx (Tensor): of size (T x n), the gradients with respect to x
        dzdw (Tensor): of size (n x m), the gradients with respect to w
        dzdb (Tensor): of size (m), the gradients with respect to b
        """
        x, w, b = ctx.saved_tensors
        # d(xW + b)/dx = W^T, so each upstream row is multiplied by W^T.
        dzdx = dz_dy @ w.t()
        # Each weight gradient accumulates input_i * upstream_j over the batch.
        dzdw = x.t() @ dz_dy
        # The same bias is added to every example, so its gradient sums over T.
        dzdb = dz_dy.sum(dim=0)

        return dzdx, dzdw, dzdb

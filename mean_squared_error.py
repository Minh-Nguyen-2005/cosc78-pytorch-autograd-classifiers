import torch


class MeanSquaredError(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        """
        computes the mean squared error between x1 (inputs) and x2 (targets)

        Arguments
        -------
        ctx: a pytorch context object
        x1: (Tensor of size (T x n) where T is the batch size and n is the number of input features.
        x2: (Tensor) of size (T x n)

        Returns
        ------
        y: (scalar) The mean squared error between x1 and x2， averaged over all T * n elements
        """
        ctx.save_for_backward(x1, x2)
        # Average over every element
        y = ((x1 - x2) ** 2).mean()

        return y

    @staticmethod
    def backward(ctx, dzdy):
        """
        back-propagates the error with respect to the input arguments

        Arguments
        --------
        ctx: A PyTorch context object
        dzdy:  a scalar (Tensor), the gradient with respect to y

        Returns
        ------
        dzdx1 (Tensor): of size(T x n), the gradients w.r.t x1
        dzdx2 (Tensor): of size(T x n), the gradients w.r.t x2
        """
        x1, x2 = ctx.saved_tensors
        # d mean((x1 - x2)^2)/dx1 = 2 * (x1 - x2) / number_of_elements.
        dzdx1 = dzdy * 2 * (x1 - x2) / x1.numel()
        # The target argument has the opposite sign because d(x1 - x2)/dx2 = -1.
        dzdx2 = -dzdx1

        return dzdx1, dzdx2

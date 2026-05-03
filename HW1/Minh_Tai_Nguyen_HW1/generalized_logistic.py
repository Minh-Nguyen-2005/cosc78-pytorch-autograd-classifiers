import torch


class GeneralizedLogistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l, u, g):
        """
        Computes the generalized logistic function

        Arguments
        ---------
        ctx: A PyTorch context object
        x: (Tensor) of size (T x n), the input features
        l, u, and g: (scalar tensors) representing the generalized logistic function parameters.

        Returns
        -------
        y: (Tensor) of size (T x n), the outputs of the generalized logistic operator

        """
        # Reuse PyTorch's numerically stable sigmoid for 1 / (1 + exp(-g*x)).
        s = torch.sigmoid(g * x)
        # Scale the sigmoid from [0, 1] to [l, u].
        y = l + (u - l) * s
        ctx.save_for_backward(x, l, u, g, s)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        """
        back-propagate the gradients with respect to the inputs

        Arguments
        ----------
        ctx: a PyTorch context object
        dzdy (Tensor): of size (T x n), the gradients with respect to the outputs y

        Returns
        -------
        dzdx (Tensor): of size (T x n), the gradients with respect to x
        dzdl, dzdu, and dzdg: the gradients with respect to the generalized logistic parameters
        """
        x, l, u, g, s = ctx.saved_tensors
        # Derivative of sigmoid(g*x) with respect to its pre-activation.
        ds = s * (1 - s)
        dzdx = dzdy * (u - l) * g * ds
        # l, u, and g are scalar parameters, so their elementwise gradients sum over x.
        dzdl = (dzdy * (1 - s)).sum().reshape_as(l)
        dzdu = (dzdy * s).sum().reshape_as(u)
        dzdg = (dzdy * (u - l) * x * ds).sum().reshape_as(g)

        return dzdx, dzdl, dzdu, dzdg

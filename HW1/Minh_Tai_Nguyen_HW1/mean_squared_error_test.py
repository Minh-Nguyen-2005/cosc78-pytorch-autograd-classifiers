from mean_squared_error import MeanSquaredError
import torch


def mean_squared_error_test():
    """
     Unit tests for the MeanSquaredError autograd Function.

    PROVIDED CONSTANTS
    ------------------
    TOL (float): the absolute error tolerance for the backward mode. If any error is equal to or
                greater than TOL, is_correct is false
    DELTA (float): The difference parameter for the finite difference computation
    X1 (Tensor): size (48 x 2) denoting 48 example inputs each with 2 features
    X2 (Tensor): size (48 x 2) denoting the targets

    Returns
    -------
    is_correct (boolean): True if and only if MeanSquaredError passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx1 (float): the  error between the analytical and numerical gradients w.r.t X1
                    2. dzdx2 (float): The error between the analytical and numerical gradients w.r.t X2
    Note
    -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%
    dataset = torch.load("mean_squared_error_test.pt")
    X1 = dataset["X1"]
    X2 = dataset["X2"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    mean_squared_error = MeanSquaredError.apply
    # %%% DO NOT EDIT ABOVE %%%

    x1 = X1.detach().clone().requires_grad_(True)
    x2 = X2.detach().clone().requires_grad_(True)

    y = mean_squared_error(x1, x2)
    dzdy = torch.randn_like(y)
    # Analytical gradients exercise the custom backward method.
    dzdx1, dzdx2 = torch.autograd.grad(y, (x1, x2), dzdy)

    with torch.no_grad():
        # Central differences estimate each partial derivative independently.
        delta = float(DELTA)
        x1_0 = X1.detach()
        x2_0 = X2.detach()
        dzdy0 = dzdy.detach()
        t, n = x1_0.shape

        x1_eye = torch.eye(x1_0.numel(), dtype=x1_0.dtype, device=x1_0.device).reshape(x1_0.numel(), t, n)
        x1_plus = x1_0.unsqueeze(0) + delta * x1_eye
        x1_minus = x1_0.unsqueeze(0) - delta * x1_eye
        y_x1_plus = ((x1_plus - x2_0.unsqueeze(0)) ** 2).mean(dim=(1, 2))
        y_x1_minus = ((x1_minus - x2_0.unsqueeze(0)) ** 2).mean(dim=(1, 2))
        dzdx1_num = (dzdy0 * (y_x1_plus - y_x1_minus) / (2 * delta)).reshape_as(x1_0)

        x2_eye = torch.eye(x2_0.numel(), dtype=x2_0.dtype, device=x2_0.device).reshape(x2_0.numel(), t, n)
        x2_plus = x2_0.unsqueeze(0) + delta * x2_eye
        x2_minus = x2_0.unsqueeze(0) - delta * x2_eye
        y_x2_plus = ((x1_0.unsqueeze(0) - x2_plus) ** 2).mean(dim=(1, 2))
        y_x2_minus = ((x1_0.unsqueeze(0) - x2_minus) ** 2).mean(dim=(1, 2))
        dzdx2_num = (dzdy0 * (y_x2_plus - y_x2_minus) / (2 * delta)).reshape_as(x2_0)

    err = {
        "dzdx1": (dzdx1 - dzdx1_num).abs().max(),
        "dzdx2": (dzdx2 - dzdx2_num).abs().max()
    }

    gradcheck_correct = torch.autograd.gradcheck(
        mean_squared_error,
        (
            X1.detach().clone().requires_grad_(True),
            X2.detach().clone().requires_grad_(True)
        ),
        eps=float(DELTA),
        atol=float(TOL)
    )
    # The test passes only if gradcheck and both numerical comparisons pass.
    is_correct = gradcheck_correct and all(error < TOL for error in err.values())
    err = {key: value.item() for key, value in err.items()}
    torch.save([is_correct, err], "mean_squared_error_test_results.pt")

    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = mean_squared_error_test()
    assert tests_passed
    print(errors)

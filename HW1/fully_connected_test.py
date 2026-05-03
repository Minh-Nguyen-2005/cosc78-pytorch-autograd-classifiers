from fully_connected import FullyConnected
import torch


def fully_connected_test():
    """
    Provides Unit tests for the FullyConnected autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL (float): The error tolerance for the backward mode. If the error >= TOL, then is_correct is false
    DELTA (float): The difference parameter for the finite difference computations
    X (Tensor): of size (48 x 2), the inputs
    W (Tensor): of size (2 x 72), the weights
    B (Tensor): of size (72), the biases

    Returns
    -------
    is_correct (boolean): True if and only iff FullyConnected passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx: the error between the analytical and numerical gradients w.r.t X
                    2. dzdw (float): ... w.r.t W
                    3. dzdb (float): ... w.r.t B

    Note
    ----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%%
    dataset = torch.load("fully_connected_test.pt")
    X = dataset["X"]
    W = dataset["W"]
    B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    full_connected = FullyConnected.apply
    # %%% DO NOT EDIT ABOVE

    x = X.detach().clone().requires_grad_(True)
    w = W.detach().clone().requires_grad_(True)
    b = B.detach().clone().requires_grad_(True)

    y = full_connected(x, w, b)
    dzdy = torch.randn_like(y)
    # Analytical gradients come from our custom backward implementation.
    dzdx, dzdw, dzdb = torch.autograd.grad(y, (x, w, b), dzdy)

    with torch.no_grad():
        # Numerical gradients use central differences while autograd is disabled.
        delta = float(DELTA)
        x0 = X.detach()
        w0 = W.detach()
        b0 = B.detach()
        dzdy0 = dzdy.detach()
        t, n = x0.shape
        m = b0.shape[0]

        x_eye = torch.eye(x0.numel(), dtype=x0.dtype, device=x0.device).reshape(x0.numel(), t, n)
        x_plus = x0.unsqueeze(0) + delta * x_eye
        x_minus = x0.unsqueeze(0) - delta * x_eye
        y_x_plus = full_connected(x_plus.reshape(x0.numel() * t, n), w0, b0).reshape(x0.numel(), t, m)
        y_x_minus = full_connected(x_minus.reshape(x0.numel() * t, n), w0, b0).reshape(x0.numel(), t, m)
        dzdx_num = (((y_x_plus - y_x_minus) / (2 * delta)) * dzdy0.unsqueeze(0)).sum(dim=(1, 2)).reshape_as(x0)

        w_eye = torch.eye(w0.numel(), dtype=w0.dtype, device=w0.device).reshape(w0.numel(), n, m)
        w_plus = w0.unsqueeze(0) + delta * w_eye
        w_minus = w0.unsqueeze(0) - delta * w_eye
        y_w_plus = full_connected(
            x0,
            w_plus.permute(1, 0, 2).reshape(n, w0.numel() * m),
            b0.repeat(w0.numel())
        ).reshape(t, w0.numel(), m).permute(1, 0, 2)
        y_w_minus = full_connected(
            x0,
            w_minus.permute(1, 0, 2).reshape(n, w0.numel() * m),
            b0.repeat(w0.numel())
        ).reshape(t, w0.numel(), m).permute(1, 0, 2)
        dzdw_num = (((y_w_plus - y_w_minus) / (2 * delta)) * dzdy0.unsqueeze(0)).sum(dim=(1, 2)).reshape_as(w0)

        b_eye = torch.eye(b0.numel(), dtype=b0.dtype, device=b0.device)
        b_plus = b0.unsqueeze(0) + delta * b_eye
        b_minus = b0.unsqueeze(0) - delta * b_eye
        w_repeated = w0.unsqueeze(0).expand(b0.numel(), n, m).permute(1, 0, 2).reshape(n, b0.numel() * m)
        y_b_plus = full_connected(x0, w_repeated, b_plus.reshape(b0.numel() * m)).reshape(t, b0.numel(), m).permute(1, 0, 2)
        y_b_minus = full_connected(x0, w_repeated, b_minus.reshape(b0.numel() * m)).reshape(t, b0.numel(), m).permute(1, 0, 2)
        dzdb_num = (((y_b_plus - y_b_minus) / (2 * delta)) * dzdy0.unsqueeze(0)).sum(dim=(1, 2)).reshape_as(b0)

    err = {
        "dzdx": (dzdx - dzdx_num).abs().max(),
        "dzdw": (dzdw - dzdw_num).abs().max(),
        "dzdb": (dzdb - dzdb_num).abs().max()
    }

    gradcheck_correct = torch.autograd.gradcheck(
        full_connected,
        (
            X.detach().clone().requires_grad_(True),
            W.detach().clone().requires_grad_(True),
            B.detach().clone().requires_grad_(True)
        ),
        eps=float(DELTA),
        atol=float(TOL)
    )
    # Passing requires finite differences, gradcheck, and all errors below tolerance.
    is_correct = gradcheck_correct and all(error < TOL for error in err.values())
    err = {key: value.item() for key, value in err.items()}
    torch.save([is_correct, err], "fully_connected_test_results.pt")

    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    assert tests_passed
    print(errors)

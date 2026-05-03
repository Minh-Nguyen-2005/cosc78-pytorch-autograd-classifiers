from generalized_logistic import GeneralizedLogistic
import torch


def generalized_logistic_test():
    """
    Provides Unit tests for the GeneralizedLogistic autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL1 (float): the  error tolerance for the forward mode. If the error >= TOL1, is_correct is false
    TOL2 (float): The error tolerance for the backward mode
    DELTA (float): The difference parameter for the finite differences computation
    X (Tensor): size (48 x 2) of inputs
    L, U, and G (floats): The parameter values necessary to compute the hyperbolic tangent (tanH) using
                        GeneralizedLogistic
    Returns:
    -------
    is_correct (boolean): True if and only if GeneralizedLogistic passes all unit tests
    err (Dictionary): with the following keys
                        1. y (float): The error between the forward direction and the results of pytorch's tanH
                        2. dzdx (float): the error between the analytical and numerical gradients w.r.t X
                        3. dzdl (float): ... w.r.t L
                        4. dzdu (float): ... w.r.t U
                        5. dzdg (float): .. w.r.t G
     Note
     -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%%% DO NOT EDIT BELOW %%%
    dataset = torch.load("generalized_logistic_test.pt")
    X = dataset["X"]
    L = dataset["L"]
    U = dataset["U"]
    G = dataset["G"]
    TOL1 = dataset["TOL1"]
    TOL2 = dataset["TOL2"]
    DELTA = dataset["DELTA"]
    generalized_logistic = GeneralizedLogistic.apply
    # %%%  DO NOT EDIT ABOVE %%%

    x = X.detach().clone().requires_grad_(True)
    l = L.detach().clone().requires_grad_(True)
    u = U.detach().clone().requires_grad_(True)
    g = G.detach().clone().requires_grad_(True)

    y = generalized_logistic(x, l, u, g)
    # Use J = mean(y) as the scalar objective for a stable finite-difference test.
    dzdy = torch.ones_like(y) / y.numel()
    dzdx, dzdl, dzdu, dzdg = torch.autograd.grad(y, (x, l, u, g), dzdy)

    with torch.no_grad():
        # These constants define tanH as a special case of generalized logistic.
        delta = float(DELTA)
        x0 = X.detach()
        l0 = L.detach()
        u0 = U.detach()
        g0 = G.detach()
        dzdy0 = dzdy.detach()
        t, n = x0.shape

        err_y = (generalized_logistic(x0, l0, u0, g0) - torch.tanh(x0)).abs().max()

        x_eye = torch.eye(x0.numel(), dtype=x0.dtype, device=x0.device).reshape(x0.numel(), t, n)
        x_plus = x0.unsqueeze(0) + delta * x_eye
        x_minus = x0.unsqueeze(0) - delta * x_eye
        y_x_plus = generalized_logistic(x_plus.reshape(x0.numel() * t, n), l0, u0, g0).reshape(x0.numel(), t, n)
        y_x_minus = generalized_logistic(x_minus.reshape(x0.numel() * t, n), l0, u0, g0).reshape(x0.numel(), t, n)
        dzdx_num = (((y_x_plus - y_x_minus) / (2 * delta)) * dzdy0.unsqueeze(0)).sum(dim=(1, 2)).reshape_as(x0)

        y_l_plus = generalized_logistic(x0, l0 + delta, u0, g0)
        y_l_minus = generalized_logistic(x0, l0 - delta, u0, g0)
        dzdl_num = (((y_l_plus - y_l_minus) / (2 * delta)) * dzdy0).sum().reshape_as(l0)

        y_u_plus = generalized_logistic(x0, l0, u0 + delta, g0)
        y_u_minus = generalized_logistic(x0, l0, u0 - delta, g0)
        dzdu_num = (((y_u_plus - y_u_minus) / (2 * delta)) * dzdy0).sum().reshape_as(u0)

        y_g_plus = generalized_logistic(x0, l0, u0, g0 + delta)
        y_g_minus = generalized_logistic(x0, l0, u0, g0 - delta)
        dzdg_num = (((y_g_plus - y_g_minus) / (2 * delta)) * dzdy0).sum().reshape_as(g0)

    err = {
        "dzdx": (dzdx - dzdx_num).abs().max(),
        "dzdl": (dzdl - dzdl_num).abs().max(),
        "dzdu": (dzdu - dzdu_num).abs().max(),
        "dzdg": (dzdg - dzdg_num).abs().max(),
        "y": err_y
    }

    gradcheck_correct = torch.autograd.gradcheck(
        generalized_logistic,
        (
            X.detach().clone().requires_grad_(True),
            L.detach().clone().requires_grad_(True),
            U.detach().clone().requires_grad_(True),
            G.detach().clone().requires_grad_(True)
        ),
        eps=float(DELTA),
        atol=float(TOL2)
    )
    # Forward tanH error uses TOL1; all backward checks use TOL2.
    is_correct = gradcheck_correct and err["y"] < TOL1 and all(err[key] < TOL2 for key in ["dzdx", "dzdl", "dzdu", "dzdg"])
    err = {key: value.item() for key, value in err.items()}
    torch.save([is_correct, err], "generalized_logistic_test_results.pt")

    return is_correct, err


if __name__ == '__main__':
    test_passed, errors = generalized_logistic_test()
    assert test_passed
    print(errors)

import torch


def _mask_observed(M_mask: torch.Tensor, i: int, t: int):
    return torch.nonzero(M_mask[i, t], as_tuple=False).squeeze(1)


def factor_E_step(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
):
    """
    Compute posterior means of factors and log-likelihood contribution.

    Args:
        Y: (I, T, J) residuals.
        M_mask: bool mask (I, T, J) True=observed.
        L: loadings (J, r).
        psi: diag entries of Psi, shape (J,).
        s: subject scales (I,).
        a2: factor variances (I, T, r).

    Returns:
        F_tilde: (I, T, r) posterior means.
        ll: scalar log-likelihood.
    """
    I, T, J = Y.shape
    r = L.shape[1]
    device = Y.device
    F_tilde = torch.zeros((I, T, r), device=device)
    ll = torch.zeros((), device=device)

    Psi_diag = psi
    for i in range(I):
        Psi_i = torch.diag(Psi_diag * (s[i] ** 2))
        for t in range(T):
            obs_idx = _mask_observed(M_mask, i, t)
            if obs_idx.numel() == 0:
                continue
            y_obs = Y[i, t, obs_idx]
            L_obs = L[obs_idx]
            Psi_obs = Psi_i[obs_idx][:, obs_idx]
            A_it = torch.diag(a2[i, t])

            H_obs = L_obs @ A_it @ L_obs.T + Psi_obs
            H_inv_y = torch.linalg.solve(H_obs, y_obs)
            F_mean = A_it @ L_obs.T @ H_inv_y
            F_tilde[i, t] = F_mean

            sign, logdet = torch.linalg.slogdet(H_obs)
            quad = y_obs @ H_inv_y
            ll = ll - 0.5 * (logdet + quad)
    return F_tilde, ll


def factor_log_likelihood(
    Y: torch.Tensor,
    M_mask: torch.Tensor,
    L: torch.Tensor,
    psi: torch.Tensor,
    s: torch.Tensor,
    a2: torch.Tensor,
):
    """
    Same as factor_E_step but returns only log-likelihood (differentiable).
    """
    _, ll = factor_E_step(Y, M_mask, L, psi, s, a2)
    return ll


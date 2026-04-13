import numpy as np
from scipy.interpolate import BSpline
import sparse
from scipy.interpolate._bspl import evaluate_all_bspl
try:
        import torch
        from torch import nn
except Exception:  # torch is optional for torch-based variants
        torch = None
        
def generate_data(n_patients, n_var, T, x_coef, idx_x, idx_y, rank, k, N):
        D = N-k
        Fx = np.random.randn(n_var,rank)
        Fy = np.random.randn(rank)

        knots = np.array(list(range(1,N-2*k-1)))/(N-2*k-1)*T
        knots = np.insert(knots, 0, (k+1)*[-1])
        knots = np.insert(knots, N-k-1, (k+1)*[T+1])

        weights = np.random.randn(n_patients, rank, D)

        x_data = []
        y_data = []
        for i in range(n_patients):
            # Setting 1.1
            spl = [lambda t: 0.02*i*np.log(t+1), lambda t: 2*np.exp(-(t-60+10*i)/50*(t-60+10*i)+0.0000001) + 4*np.exp(-(t-70+10*i)/20*(t-70+10*i)+0.0000001) , lambda t: np.cos(0.12*np.pi*t) + 1]
           # Setting 1.2
           # spl = [lambda t: 0.02*np.log(t+1), lambda t: 2*np.exp(-(t-60+10*i)/50*(t-60+10*i)+0.0000001) + 4*np.exp(-(t-70+10*i)/20*(t-70+10*i)+0.0000001) , lambda t: np.cos(0.12*np.pi*t) + 1]
            y_tmp=np.zeros((T))
            for j in range(n_var):
                tmp = np.matmul(Fx[j,:], [spl[r](idx_x[i,j,:].data) for r in range(rank)])+ 0.5*np.random.randn(len(idx_x[i,j,:].data))
                x_data = np.concatenate((x_data, tmp))
                y_tmp+= x_coef[j]*tmp
            y_data = np.concatenate((y_data, y_tmp))
        print(len(x_data))
        output_x = sparse.COO(idx_x.coords, x_data, shape = (n_patients, n_var, T))
        output_y = sparse.COO(idx_y.coords, y_data, shape = (n_patients, 1, T))

        return [output_x, output_y, knots, weights, Fx, Fy]

def xy_pred(weights, knots, F, n_patients, n_var, idx, rank, k):
        thetas = [[BSpline(knots, weights[i,r,:], k) for r in range(rank)] for i in range(n_patients)]
        data = []
        for i in range(n_patients):
            for j in range(n_var+1):
                tmp = np.matmul(F[j,:],[thetas[i][r](idx[i,j,:].data) for r in range(rank)])
                data = np.concatenate((data, tmp))
        output = sparse.COO(idx.coords, data,shape = (n_patients, n_var+1, idx.shape[2]))
        return output

def y_pred(weights, knots, F, n_patients, n_var, idx_y, rank, k):
        thetas = [[BSpline(knots, weights[i,r,:], k) for r in range(rank)] for i in range(n_patients)]
        data = []
        for i in range(n_patients):
            #for j in range(n_var+1):
                tmp = np.matmul(F[n_var,:],[thetas[i][r](idx_y[i,0,:].data) for r in range(rank)])
                data = np.concatenate((data, tmp))
        output = sparse.COO(idx_y.coords, data,shape = (n_patients, 1, idx_y.shape[2]))
        return output

def dlts(X, Y, n_patients, n_var, T, idx_x, idx_y, rank, k, N, lambda1 = 1, lambda2 = 1, Niter = 100, alpha = 0.001, ebs = 0.0001, l=1):
        beta1 = 0.9
        beta2 = 0.999
        m_w = 0
        m_F = 0
        v_w = 0
        v_F = 0

        D = N-k-1
        F = np.random.randn(n_var+1,rank)


        coords = np.copy(Y.coords)
        coords[1,:] = n_var
        coords = np.concatenate((X.coords,coords), axis = 1)
        data = np.concatenate((X.data,Y.data))
        xy = sparse.COO(coords, data, shape = (n_patients, n_var+1, T))


        knots = np.array(list(range(1,N-2*k-1)))/(N-2*k-1)*T
        knots = np.insert(knots, 0, (k+1)*[-1])
        knots = np.insert(knots, N-k-1, (k+1)*[T+1])

        coords = np.copy(idx_y.coords)
        coords[1,:] = n_var
        coords = np.concatenate((idx_x.coords,coords), axis = 1)
        data = np.concatenate((idx_x.data,idx_y.data))
        idx = sparse.COO(coords, data, shape = (n_patients, n_var+1, T))

        weights = np.random.randn(n_patients, rank, D)
        xy_hat = xy_pred(weights, knots, F, n_patients, n_var, idx, rank, k)


        nobs = len(data)
        S = np.sum((xy - xy_hat)**2)/nobs
        S_record = [S]
        print(S)

        K = np.zeros((D, T), dtype=np.float_)
        for t in range(T):
            xval = t
            if xval <= knots[k]:
                left = k
            else:
                left = np.searchsorted(knots, xval) - 1

            # fill a row
            bb = evaluate_all_bspl(knots*1.0, k, xval, left)
            K[left-k:left+1, t] = bb

        for itr in range(Niter):
            unique_t = np.sort(list(set(idx.data)))
            theta = np.tensordot(weights, K, axes = (2,0))
            grad_weights = np.tensordot(2*(xy_hat - xy), F, axes = (1, 0))
            grad_weights = np.tensordot(grad_weights, K, axes = (1, 1))
            ## Fused lasso for theta_i
            grad_pen = np.empty(weights.shape)
            trans_m = -np.eye(n_patients- 1)
            for i in range(n_patients):
                tmp = np.tensordot(np.insert(trans_m, i, np.ones(n_patients-1), axis=1), weights, axes = (1,0))
                tmp = np.sign(tmp)
                grad_pen[i] = np.sum(tmp, axis = 0)
            ##total variation for bspline
            jump = np.insert(-np.eye(D-1), 0, np.zeros(D-1), axis = 1)
            jump = np.insert(jump, D-1, np.zeros(D), axis = 0)
            jump = jump + np.eye(D)
            jump_m = jump
            if k > 1:
                for i in range(k):
                    jump_m = np.matmul(jump_m, jump)
            grad_pen1 = np.tensordot(weights, jump_m.T, axes = (2, 0))
            grad_pen1 = np.sign(grad_pen1)
            grad_pen1 = np.tensordot(grad_pen1[:,:,0:(D-k)], jump_m[0:(D-k)], axes = (2, 0))
            grad_weights = grad_weights + l*weights + lambda1 * grad_pen + lambda2 * grad_pen1
            grad_F = np.tensordot(2*(xy_hat - xy),theta, axes = ([0,2],[0,2]))
            grad_F += l*F

            m_w = beta1*m_w + (1-beta1) * grad_weights
            m_F = beta1*m_F + (1-beta1) * grad_F
            v_w = beta2 * v_w + (1-beta2) * grad_weights**2
            v_F = beta2 * v_F + (1-beta2) * grad_F**2
            mhat_w = m_w / (1-beta1)
            mhat_F = m_F / (1 - beta1)
            vhat_w = v_w / (1 - beta2)
            vhat_F = v_F / (1 - beta2)
            beta1 = beta1**(i+1)
            beta2 = beta2**(i+1)

            weights = weights - alpha * mhat_w /(np.sqrt(vhat_w) + 1e-8)
            F = F - alpha * mhat_F /(np.sqrt(vhat_F) + 1e-8)

            xy_hat = xy_pred(weights, knots, F, n_patients, n_var, idx, rank, k)
            S = np.sum((xy -xy_hat)**2)/nobs
            t = np.abs((S_record[-1] - S)/S_record[-1])
            if i > 10 and S >= np.max(S_record):
                print('Diverge')
                break
            if t < ebs:
                print(itr)
                print('Converge')
                S_record.append(S)
                break
            if itr%100 == 0:
                print(itr, S)

        print('Max iteration')
        X_hat = xy_hat[:, 0:n_var, :]
        Y_hat = xy_hat[:, n_var, :]
        Y_hat = Y_hat.reshape((n_patients, 1, T))

        return [weights, F, X_hat, Y_hat]


# =============================
# NumPy-based variants
# =============================

def _bspline_basis_matrix_numpy(knots, k, T):
        """
        Build B-spline basis matrix K (D x T) using evaluate_all_bspl.
        D = N - k - 1 implied by knots length.
        """
        # D can be inferred by counting inner basis functions; here follow original fill-in logic
        # Build K by scanning t=0..T-1
        # Estimate D as max index written during fill; initialize generously then trim
        # Safer: deduce D from knots: D = len(knots) - k - 1
        D = len(knots) - k - 1
        K = np.zeros((D, T), dtype=np.float_)
        for t in range(T):
                xval = t
                if xval <= knots[k]:
                        left = k
                else:
                        left = np.searchsorted(knots, xval) - 1
                bb = evaluate_all_bspl(knots*1.0, k, xval, left)
                K[left-k:left+1, t] = bb
        return K


def xy_pred_numpy(weights, knots, F, n_patients, n_var, idx, rank, k):
        """
        NumPy vectorized prediction on irregular grid defined by idx (sparse.COO coords).
        Returns sparse.COO aligned to idx.
        """
        K = _bspline_basis_matrix_numpy(knots, k, idx.shape[2])  # (D,T)
        # theta: (n_patients, rank, T)
        theta = np.tensordot(weights, K, axes=(2, 0))
        coords = idx.coords  # (3, Kobs)
        i_arr = coords[0]
        j_arr = coords[1]
        t_arr = coords[2]
        # Gather theta[i, :, t] -> (Kobs, rank)
        Theta_eval = theta[i_arr, :, t_arr]
        # yhat = sum_r F[j, r] * theta[i, r, t]
        yhat = np.sum(F[j_arr, :] * Theta_eval, axis=1)
        return sparse.COO(coords, yhat, shape=(n_patients, n_var+1, idx.shape[2]))


def y_pred_numpy(weights, knots, F, n_patients, n_var, idx_y, rank, k):
        """
        NumPy variant for Y stream only (j = n_var).
        """
        # Reuse xy_pred_numpy but restrict coords
        K = _bspline_basis_matrix_numpy(knots, k, idx_y.shape[2])
        theta = np.tensordot(weights, K, axes=(2, 0))
        coords = idx_y.coords
        i_arr = coords[0]
        t_arr = coords[2]
        Theta_eval = theta[i_arr, :, t_arr]
        yhat = np.sum(F[n_var, :] * Theta_eval, axis=1)
        return sparse.COO(coords, yhat, shape=(n_patients, 1, idx_y.shape[2]))


def dlts_numpy(X, Y, n_patients, n_var, T, idx_x, idx_y, rank, k, N, lambda1 = 1, lambda2 = 1, Niter = 100, alpha = 0.001, ebs = 0.0001, l=1):
        """
        NumPy-only reimplementation: vectorized basis, coordinate-wise residuals, Adam updates.
        Keeps the original objective structure.
        """
        # Setup
        D = N - k - 1
        F = np.random.randn(n_var+1, rank)
        # Build combined sparse xy and index tensors
        coords = np.copy(Y.coords)
        coords[1, :] = n_var
        coords = np.concatenate((X.coords, coords), axis=1)
        data = np.concatenate((X.data, Y.data))
        xy = sparse.COO(coords, data, shape=(n_patients, n_var+1, T))

        coords_idx = np.copy(idx_y.coords)
        coords_idx[1, :] = n_var
        coords_idx = np.concatenate((idx_x.coords, coords_idx), axis=1)
        data_idx = np.concatenate((idx_x.data, idx_y.data))
        idx = sparse.COO(coords_idx, data_idx, shape=(n_patients, n_var+1, T))

        weights = np.random.randn(n_patients, rank, D)

        # Basis matrix
        knots = np.array(list(range(1, N - 2 * k - 1))) / (N - 2 * k - 1) * T
        knots = np.insert(knots, 0, (k + 1) * [-1])
        knots = np.insert(knots, N - k - 1, (k + 1) * [T + 1])
        K = _bspline_basis_matrix_numpy(knots, k, T)  # (D,T)

        # Adam buffers
        beta1, beta2 = 0.9, 0.999
        m_w = np.zeros_like(weights)
        v_w = np.zeros_like(weights)
        m_F = np.zeros_like(F)
        v_F = np.zeros_like(F)
        t_step = 0

        # Prepare coordinate arrays for residual computation
        c = idx.coords
        i_arr, j_arr, t_arr = c[0], c[1], c[2]
        y_obs = xy.data  # aligned with idx order
        nobs = y_obs.size

        # Helper to compute residuals and predictions
        def compute_pred_and_res(weights, F):
                theta = np.tensordot(weights, K, axes=(2, 0))  # (I,R,T)
                Theta_eval = theta[i_arr, :, t_arr]            # (Kobs,R)
                yhat = np.sum(F[j_arr, :] * Theta_eval, axis=1)
                res = yhat - y_obs
                return theta, Theta_eval, yhat, res

        # Initial loss
        _, _, _, res = compute_pred_and_res(weights, F)
        S = float(np.mean(res ** 2))
        S_record = [S]

        # Precompute TV operator for spline dimension (first order; extendable to k-order)
        # jump_m approximates original jump^k
        J = np.eye(D)
        Jm = J.copy()
        diff = np.eye(D) - np.roll(np.eye(D), 1, axis=1)
        # approximate k-th order via repeated multiplication
        for _ in range(max(1, k)):
                Jm = Jm @ diff

        for itr in range(Niter):
            t_step += 1
            theta, Theta_eval, yhat, res = compute_pred_and_res(weights, F)
            # Gradients
            # grad_F[j,r] = 2/n sum_{obs with stream j} res * Theta_eval[:,r]
            grad_F = np.zeros_like(F)
            for j in range(n_var + 1):
                sel = (j_arr == j)
                if sel.any():
                        grad_F[j, :] = (2.0 / nobs) * (res[sel][:, None] * Theta_eval[sel]).sum(axis=0)
            # grad_weights[i,r,d] = 2/n sum_{obs with subject i} res * sum_j F[j,r] * K[d,t]
            grad_weights = np.zeros_like(weights)
            # Accumulate over observations by batching on unique (i,t)
            uniq_it, inv = np.unique(np.stack([i_arr, t_arr], axis=1), axis=0, return_inverse=True)
            # Residual contribution grouped by (i,t) and r via F and Theta_eval
            # For each obs, contribution for r is res * F[j,r]
            contrib_r = res[:, None] * F[j_arr, :]  # (Kobs, R)
            # Sum over same (i,t)
            Rdim = contrib_r.shape[1]
            agg = np.zeros((uniq_it.shape[0], Rdim))
            np.add.at(agg, inv, contrib_r)
            # Map back to per (i,t)
            for idx_it, (ii, tt) in enumerate(uniq_it):
                grad_weights[ii, :, :] += (2.0 / nobs) * (agg[idx_it][:, None] * K[:, tt][None, :])
            # Penalties
            # Ridge
            grad_weights += l * weights
            grad_F += l * F
            # Fused lasso across patients (adjacent differences)
            diff_pat = np.diff(weights, axis=0)
            grad_fused = np.zeros_like(weights)
            s = np.sign(diff_pat)
            grad_fused[:-1] -= s
            grad_fused[1:] += s
            grad_weights += lambda1 * grad_fused
            # TV along spline coeffs (approx k-th order by repeated diff)
            WJ = np.tensordot(weights, Jm.T, axes=(2, 0))
            grad_tv = np.sign(WJ)
            grad_tv = np.tensordot(grad_tv, Jm, axes=(2, 0))
            grad_weights += lambda2 * grad_tv

            # Adam update
            m_w = beta1 * m_w + (1 - beta1) * grad_weights
            v_w = beta2 * v_w + (1 - beta2) * (grad_weights ** 2)
            m_F = beta1 * m_F + (1 - beta1) * grad_F
            v_F = beta2 * v_F + (1 - beta2) * (grad_F ** 2)
            m_w_hat = m_w / (1 - beta1 ** t_step)
            v_w_hat = v_w / (1 - beta2 ** t_step)
            m_F_hat = m_F / (1 - beta1 ** t_step)
            v_F_hat = v_F / (1 - beta2 ** t_step)
            weights = weights - alpha * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
            F = F - alpha * m_F_hat / (np.sqrt(v_F_hat) + 1e-8)

            # Loss and stopping
            _, _, _, res = compute_pred_and_res(weights, F)
            S = float(np.mean(res ** 2))
            if itr > 0 and abs(S_record[-1] - S) / (S_record[-1] + 1e-12) < ebs:
                break
            S_record.append(S)

        # Outputs on full grid (dense prediction for convenience)
        theta = np.tensordot(weights, K, axes=(2, 0))  # (I,R,T)
        X_hat = np.zeros((n_patients, n_var, T))
        for j in range(n_var):
                X_hat[:, j, :] = np.einsum('irt,r->it', theta, F[j, :])
        Y_hat = np.einsum('irt,r->it', theta, F[n_var, :])[:, None, :]
        return [weights, F, X_hat, Y_hat]


# =============================
# PyTorch-based variants
# =============================

def _require_torch():
        if torch is None:
            raise ImportError("PyTorch is not available. Install torch to use torch-based functions.")


def xy_pred_torch(weights, knots, F, n_patients, n_var, idx, k, device=None):
        """
        Torch variant: weights (I,R,D), F ((n_var+1),R); returns sparse.COO with predictions.
        """
        _require_torch()
        T = idx.shape[2]
        K_np = _bspline_basis_matrix_numpy(knots, k, T)
        dev = device or (torch.device('cuda') if torch and torch.cuda.is_available() else torch.device('cpu'))
        W = torch.tensor(weights, dtype=torch.float32, device=dev)
        F_t = torch.tensor(F, dtype=torch.float32, device=dev)
        K = torch.tensor(K_np, dtype=torch.float32, device=dev)  # (D,T)
        theta = torch.tensordot(W, K, dims=([2], [0]))  # (I,R,T)
        coords = idx.coords
        i_arr = torch.tensor(coords[0], dtype=torch.long, device=dev)
        j_arr = torch.tensor(coords[1], dtype=torch.long, device=dev)
        t_arr = torch.tensor(coords[2], dtype=torch.long, device=dev)
        Theta_eval = theta[i_arr, :, t_arr]  # (Kobs,R)
        yhat = (F_t[j_arr, :] * Theta_eval).sum(dim=1).detach().cpu().numpy()
        return sparse.COO(coords, yhat, shape=(n_patients, n_var+1, T))


def y_pred_torch(weights, knots, F, n_patients, n_var, idx_y, k, device=None):
        _require_torch()
        T = idx_y.shape[2]
        K_np = _bspline_basis_matrix_numpy(knots, k, T)
        dev = device or (torch.device('cuda') if torch and torch.cuda.is_available() else torch.device('cpu'))
        W = torch.tensor(weights, dtype=torch.float32, device=dev)
        F_t = torch.tensor(F, dtype=torch.float32, device=dev)
        K = torch.tensor(K_np, dtype=torch.float32, device=dev)
        theta = torch.tensordot(W, K, dims=([2], [0]))  # (I,R,T)
        coords = idx_y.coords
        i_arr = torch.tensor(coords[0], dtype=torch.long, device=dev)
        t_arr = torch.tensor(coords[2], dtype=torch.long, device=dev)
        Theta_eval = theta[i_arr, :, t_arr]
        yhat = (F_t[n_var, :] * Theta_eval).sum(dim=1).detach().cpu().numpy()
        return sparse.COO(coords, yhat, shape=(n_patients, 1, T))


def dlts_torch(X, Y, n_patients, n_var, T, rank, k, N, lambda1 = 1, lambda2 = 1, Niter = 100, alpha = 0.001, ebs = 0.0001, l=1, device=None):
        """
        Torch reimplementation with autograd. Uses Adam and nonsmooth L1 penalties via |.|.

        Inputs X and Y can be either:
          - pydata.sparse COO tensors with observed entries, or
          - numpy arrays with shape X: (I, n_var, T), Y: (I, 1, T), where NaN indicates missing.
        """
        _require_torch()
        dev = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        D = N - k - 1
        # Build knots and K
        knots = np.array(list(range(1, N - 2 * k - 1))) / (N - 2 * k - 1) * T
        knots = np.insert(knots, 0, (k + 1) * [-1])
        knots = np.insert(knots, N - k - 1, (k + 1) * [T + 1])
        K_np = _bspline_basis_matrix_numpy(knots, k, T)
        K = torch.tensor(K_np, dtype=torch.float32, device=dev)

        # Parameters
        W = torch.nn.Parameter(torch.randn(n_patients, rank, D, device=dev))
        F = torch.nn.Parameter(torch.randn(n_var + 1, rank, device=dev))
        opt = torch.optim.Adam([W, F], lr=alpha)

        if D <= 1:
            jump_m_np = np.eye(D, dtype=np.float32)
        else:
            jump = np.insert(-np.eye(D - 1, dtype=np.float32),
                             0, np.zeros(D - 1, dtype=np.float32), axis=1)
            jump = np.insert(jump, D - 1, np.zeros(D, dtype=np.float32), axis=0)
            jump = jump + np.eye(D, dtype=np.float32)  # same as manual

            jump_m_np = jump.copy()
            if k > 1:
                for _ in range(k):
                    jump_m_np = jump_m_np @ jump  # jump^{k+1}
        jump_m = torch.tensor(jump_m_np, dtype=torch.float32, device=dev)

        # Prepare observed structure either from sparse.COO or numpy with NaNs
        use_sparse = hasattr(X, 'coords') and hasattr(Y, 'coords')
        if use_sparse:
            coords = np.copy(Y.coords)
            coords[1, :] = n_var
            coords = np.concatenate((X.coords, coords), axis=1)
            data = np.concatenate((X.data, Y.data))
            c = coords
            i_arr = torch.tensor(c[0], dtype=torch.long, device=dev)
            j_arr = torch.tensor(c[1], dtype=torch.long, device=dev)
            t_arr = torch.tensor(c[2], dtype=torch.long, device=dev)
            y_obs = torch.tensor(data, dtype=torch.float32, device=dev)
            nobs = float(y_obs.numel())
        else:
            # Expect numpy arrays with NaNs for missing
            X_np = np.asarray(X)
            Y_np = np.asarray(Y)
            if X_np.shape != (n_patients, n_var, T) or Y_np.shape != (n_patients, 1, T):
                raise ValueError("For numpy inputs, X must be (I,n_var,T) and Y must be (I,1,T)")
            maskX = ~np.isnan(X_np)
            maskY = ~np.isnan(Y_np)
            X_t = torch.tensor(np.nan_to_num(X_np, nan=0.0), dtype=torch.float32, device=dev)
            Y_t = torch.tensor(np.nan_to_num(Y_np, nan=0.0), dtype=torch.float32, device=dev)
            maskX_t = torch.tensor(maskX, dtype=torch.bool, device=dev)
            maskY_t = torch.tensor(maskY, dtype=torch.bool, device=dev)
            nobs = float(maskX_t.sum().item() + maskY_t.sum().item())

        # TV along spline coeffs: matches manual grad_pen1
        # P_tv(W) = sum_{i,r} sum_{d=0}^{D-k-1} | (jump_m @ W_{i,r})_d |
        def tv_kth(W: torch.Tensor) -> torch.Tensor:
            # W: (I, R, D)
            if D - k <= 0:
                return W.new_tensor(0.0)
            # (I,R,D) x (D,D) -> (I,R,D): row-wise multiplication by jump_m
            z = torch.matmul(W, jump_m.t())          # z_{i,r,:} = (jump_m @ W_{i,r})^T
            return z[..., :D - k].abs().sum()

        # Fused lasso across patients: all-pair L1, like manual grad_pen
        # P_fused(W) = 0.5 * sum_{i,j} ||W_i - W_j||_1 = sum_{i<j} ||W_i - W_j||_1
        def fused_patients(W: torch.Tensor) -> torch.Tensor:
            # W: (I, R, D)
            diff = W.unsqueeze(0) - W.unsqueeze(1)    # (I,I,R,D)
            return 0.5 * diff.abs().sum()

        prev_S = None
        for itr in range(Niter):
            opt.zero_grad()
            theta = torch.tensordot(W, K, dims=([2], [0]))  # (I,R,T)
            if use_sparse:
                Theta_eval = theta[i_arr, :, t_arr]
                yhat_obs = (F[j_arr, :] * Theta_eval).sum(dim=1)
                mse = ((yhat_obs - y_obs) ** 2).mean()
            else:
                # Full-grid predictions, mask to observed entries
                # pred: (I, n_var+1, T)
                pred = torch.einsum('irt,jr->ijt', theta, F)
                resX = pred[:, :n_var, :] - X_t
                resY = pred[:, n_var:n_var+1, :] - Y_t
                sqsum = (resX[maskX_t].pow(2).sum() + resY[maskY_t].pow(2).sum())
                mse = sqsum / max(1.0, nobs)
            ridge = 0.5 * l * (W.square().sum() + F.square().sum())
            fused = lambda1 * fused_patients(W)
            tv = lambda2 * tv_kth(W)
            loss = mse + ridge + fused + tv
            loss.backward()
            opt.step()

            S = float(mse.detach().cpu().item())
            if prev_S is not None and abs(prev_S - S) / (prev_S + 1e-12) < ebs:
                break
            prev_S = S

        # Return numpy outputs akin to original
        with torch.no_grad():
            theta = torch.tensordot(W, K, dims=([2], [0]))
            pred = torch.einsum('irt,jr->ijt', theta, F)  # (I, n_var+1, T)
            X_hat = pred[:, :n_var, :]
            Y_hat = pred[:, n_var:n_var+1, :]
        return [W.detach().cpu().numpy(), F.detach().cpu().numpy(), X_hat.detach().cpu().numpy(), Y_hat.detach().cpu().numpy()]
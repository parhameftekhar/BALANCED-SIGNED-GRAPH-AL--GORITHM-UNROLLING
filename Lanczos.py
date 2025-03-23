import torch

def lanczos_tridiag(
    matmul_closure,
    max_iter,
    dtype,
    device,
    matrix_shape,
    batch_shape=torch.Size(),
    init_vecs=None,
    num_init_vecs=1,
    tol=1e-5,
):
    """Lanczos tridiagonalization fully compatible with backpropagation."""
    if not callable(matmul_closure):
        raise RuntimeError(
            "matmul_closure should be a function callable object that multiplies a (Lazy)Tensor "
            "by a vector. Got a {} instead.".format(matmul_closure.__class__.__name__)
        )

    # Get initial probe vectors
    if init_vecs is None:
        init_vecs = torch.randn(matrix_shape[-1], num_init_vecs, dtype=dtype, device=device)
        init_vecs = init_vecs.expand(*batch_shape, matrix_shape[-1], num_init_vecs)
    num_init_vecs = init_vecs.size(-1)

    # Constants
    num_iter = min(max_iter, matrix_shape[-1])
    dim_dimension = -2

    # Initialize lists to collect q vectors and t matrix elements
    q_list = []
    alpha_list = []
    beta_list = []

    # Initial vector
    norm_init = torch.norm(init_vecs, 2, dim=dim_dimension, keepdim=True)
    q_prev = init_vecs / norm_init
    q_list.append(q_prev)

    # First iteration
    r = matmul_closure(q_prev)
    alpha = torch.sum(q_prev * r, dim=dim_dimension)
    alpha_list.append(alpha)
    r = r - alpha.unsqueeze(dim_dimension) * q_prev
    beta = torch.norm(r, 2, dim=dim_dimension)
    beta_list.append(beta)
    q_curr = r / beta.unsqueeze(dim_dimension)
    q_list.append(q_curr)


    # Main loop
    for k in range(1, num_iter):
        q_prev = q_list[-2] if k > 1 else q_list[0]
        q_curr = q_list[-1]
        beta_prev = beta_list[-1].unsqueeze(dim_dimension)

        # Compute next vector
        r = matmul_closure(q_curr) - q_prev * beta_prev
        alpha_curr = torch.sum(q_curr * r, dim=dim_dimension)
        alpha_list.append(alpha_curr)

        if k < num_iter - 1:
            r = r - alpha_curr.unsqueeze(dim_dimension) * q_curr
            # Reorthogonalization
            q_stack = torch.stack(q_list, dim=0)  # Stack all previous q vectors
            correction = torch.sum(r.unsqueeze(0) * q_stack, dim=dim_dimension, keepdim=True)
            correction = torch.sum(q_stack * correction, dim=0)
            r = r - correction
            r_norm = torch.norm(r, 2, dim=dim_dimension, keepdim=True)
            r = r / r_norm

            beta_curr = r_norm.squeeze(dim_dimension)
            beta_list.append(beta_curr)

            # Additional reorthogonalization if needed
            for _ in range(10):
                inner_products = torch.sum(q_stack * r.unsqueeze(0), dim=dim_dimension)
                if not torch.any(inner_products.abs() > tol):
                    break
                correction = torch.sum(r.unsqueeze(0) * q_stack, dim=dim_dimension, keepdim=True)
                correction = torch.sum(q_stack * correction, dim=0)
                r = r - correction
                r_norm = torch.norm(r, 2, dim=dim_dimension, keepdim=True)
                r = r / r_norm

            q_list.append(r)

            
            if torch.all(beta_curr.abs() <= 1e-6):
                break

    if len(q_list) != len(alpha_list):
        q_list = q_list[:-1]
        beta_list = beta_list[:-1]

    # Construct q_mat and t_mat
    q_mat = torch.stack(q_list, dim=0)  # Shape: num_iter x batch_shape x n x num_init_vecs
    num_iter_actual = len(q_list)

    # Construct tridiagonal T matrix
    alpha_tensor = torch.stack(alpha_list, dim=0)  # Shape: num_iter x batch_shape x num_init_vecs
    beta_tensor = torch.stack(beta_list, dim=0)    # Shape: (num_iter-1) x batch_shape x num_init_vecs
    t_mat = torch.zeros(num_iter_actual, num_iter_actual, *batch_shape, num_init_vecs, dtype=dtype, device=device)
    
    
    indices = torch.arange(num_iter_actual, device=device)
    t_mat[indices, indices] = alpha_tensor  # Diagonal
    if num_iter_actual > 1:
        off_diag_indices = indices[:-1]
        t_mat[off_diag_indices, off_diag_indices + 1] = beta_tensor
        t_mat[off_diag_indices + 1, off_diag_indices] = beta_tensor
    
    # Permute to match original output shape
    q_mat = q_mat.permute(-1, *range(1, 1 + len(batch_shape)), -2, 0).contiguous()
    t_mat = t_mat.permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous()
    
    # Squeeze if not in batch mode
    if not num_init_vecs > 1:
        q_mat = q_mat.squeeze(0)
        t_mat = t_mat.squeeze(0)

    return q_mat, t_mat





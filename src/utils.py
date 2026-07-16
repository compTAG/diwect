import torch

def sgn_star(x):
    return 1 - 2 * (x < 0).to(x.dtype)

def nan_clamp(x, min_value, max_value):
    """
    Equivalent to torch.clamp(x, min_value, max_value) except when x is NaN in which case it returns max_value.
    """
    min_value = x.new_tensor(min_value)
    max_value = x.new_tensor(max_value)
    return torch.fmax(torch.fmin(x, max_value), min_value)

def signed_angle(A, B):

    # dot product
    dot = (A * B).sum(dim=-1)
    # 2D wedge product
    wedge = A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]
    return torch.atan2(wedge, dot)

def simplex_normals_2d(vertex_coords, simplex_vertices):

    simp_coords = vertex_coords[simplex_vertices]
    edge_vectors = simp_coords.roll(-1, dims=1) - simp_coords
    edge_lengths = torch.linalg.norm(edge_vectors, dim=2, keepdim=True)
    edge_directions = edge_vectors / edge_lengths

    simp_normals = torch.stack([-edge_directions[..., 1], edge_directions[..., 0]], dim=-1)
    
    return simp_normals

def vertex_balls(v_coords, num_heights):

    v_norms = torch.linalg.norm(v_coords, dim=1)
    v_norms = v_norms.unsqueeze(1)
    max_height = torch.max(v_norms)
    zero_norms = v_norms == 0
    safe_v_norms = torch.where(zero_norms, torch.ones_like(v_norms), v_norms)

    v_centers = -v_coords / safe_v_norms
    v_centers[zero_norms.squeeze(1), 0] = -1.0

    heights = torch.linspace(-max_height, max_height, num_heights)
    normed_heights = heights / safe_v_norms

    v_radii = torch.acos(nan_clamp(-normed_heights, -1.0, 1.0))
    v_radii = torch.where(
        zero_norms,
        torch.where(heights.unsqueeze(0) < 0, torch.zeros_like(v_radii), torch.full_like(v_radii, torch.pi)),
        v_radii,
    )

    return v_centers, v_radii

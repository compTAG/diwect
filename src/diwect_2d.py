import torch
from utils import sgn_star, signed_angle, simplex_normals_2d, vertex_balls

# To compute DIWECT 2D, we measure the "contribution" of simplices of each dimension,
# and compute their alternating sums. 
def compute_simplex_measures(v_coords, v_balls, simplices):

    # Extract vertices and weights
    simp_verts, simp_weights = simplices
    simp_weights = simp_weights.unsqueeze(-1)
    v_centers, v_radii = v_balls
    print(v_balls)

    simp_normals = simplex_normals_2d(v_coords, simp_verts)
    print(simp_normals)

    simp_alphas = signed_angle(simp_normals, v_centers[simp_verts])
    simp_alphas = simp_alphas.unsqueeze(dim=-1)

    simp_betas = signed_angle(v_centers[simp_verts], simp_normals.roll(1, dims=1))
    simp_betas = simp_betas.unsqueeze(dim=-1)
    simp_cone_measures = signed_angle(simp_normals, simp_normals.roll(1, dims=1)).abs().unsqueeze(dim=-1)
    simp_radii = v_radii[simp_verts]

    simp_vert_measures = (
        sgn_star(simp_alphas) * torch.min(simp_radii, simp_alphas.abs()) +
        sgn_star(simp_betas) * torch.min(simp_radii, simp_betas.abs()) +
        .5 * simp_radii * (1 - sgn_star(simp_alphas)) * (1 - sgn_star(simp_betas))
    )
    simp_vert_measures = torch.where(
        torch.isclose(simp_radii, torch.full_like(simp_radii, torch.pi)),
        simp_cone_measures,
        simp_vert_measures,
    )
    simp_measures = torch.sum(simp_vert_measures, dim=1)
    weighted_simp_measures = simp_weights * simp_measures

    return torch.sum(weighted_simp_measures, dim=0)

def compute_diwect_2d(vertices, edges, triangles, num_heights):

    v_coords, v_weights = vertices
    v_weights = v_weights.unsqueeze(1)

    v_balls = vertex_balls(v_coords, num_heights)
    v_radii = v_balls[1]

    diwect = (
        torch.sum(2 * v_weights * v_radii, dim=0) + # contribution of the vertices
        - compute_simplex_measures(v_coords, v_balls, edges) + # contribution of the edges
        compute_simplex_measures(v_coords, v_balls, triangles) # contribution of the triangles
    )
    print(type(diwect))

    return diwect





vertices = (
    # vertex coordinates
    torch.tensor([
        [0.0, 0.0],
    ]),

    # vertex weights
    torch.tensor([
        1.0,
    ])
)

edges = (
    # edge vertex indices
    torch.empty((0, 2), dtype=torch.long),

    # edge weights
    torch.empty((0,), dtype=torch.float32)
)

triangles = (
    # triangle vertex indices
    torch.empty((0, 3), dtype=torch.long),

    # triangle weights
    torch.empty((0,), dtype=torch.float32)
)

value = compute_diwect_2d(vertices, edges, triangles, 10)
print(value)

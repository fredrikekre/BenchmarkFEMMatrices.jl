# BenchmarkFEMMatrices

Sparse matrices from finite element discretizations (using
[Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)) of various model problems, meant
as test/benchmark matrices for linear solvers and preconditioners.

Every problem is posed on `[-1, 1]^d` (`d = 2` or `3` depending on the cell type) meshed
with `nels` elements per side, and every setup function returns `(K, f)`: the sparse
system matrix and right hand side, with Dirichlet boundary conditions applied
symmetrically with `Ferrite.apply!`.

```julia
using BenchmarkFEMMatrices
using Ferrite: Triangle, Quadrilateral, Tetrahedron, Hexahedron

K, f = heat_equation(Hexahedron; nels = 50, order = 2)
K, f = convection_diffusion(Triangle; nels = 100, diffusivity = 1.0e-3)
K, f = linear_elasticity(Tetrahedron; nels = 30, ν = 0.49)
K, f = stokes(Triangle; nels = 64)
K, f = darcy(Quadrilateral; nels = 100, barrier_permeability = 1.0e-6)
K, f = dg_heat_equation(Hexahedron; nels = 40)
```

The problems are chosen to cover distinct matrix classes:

| Function | Problem | Matrix properties |
|:---|:---|:---|
| `heat_equation` | Poisson, continuous Lagrange | symmetric positive definite, M-matrix-like |
| `convection_diffusion` | Poisson + rotating wind | nonsymmetric (`diffusivity` controls how strongly) |
| `linear_elasticity` | clamped cantilever | SPD, `dim` dofs per node, rigid-body near-nullspace; `ν → 0.5` degrades conditioning |
| `stokes` | lid-driven cavity, Taylor–Hood (P2/P1 or Q2/Q1) | symmetric indefinite saddle point, zero (2,2)-block |
| `darcy` | mixed Poisson with high-contrast barrier, RT0 × P0 | symmetric indefinite saddle point, mass-matrix (1,1)-block, facet dofs |
| `dg_heat_equation` | Poisson, symmetric interior penalty DG | SPD, dense element blocks coupled across facets, no M-matrix property |

All problems accept `Triangle`/`Quadrilateral` (2d) and `Tetrahedron`/`Hexahedron` (3d),
and `nels` can be an `Int` or a tuple. See the docstrings for details and
problem-specific keyword arguments.

## Choosing `nels`: problem size, nnz, and memory

The tables below give `nels` values that result in roughly 10⁴, 10⁵, 10⁶, and 10⁷ dofs
(i.e. matrix rows, including Dirichlet-constrained ones) with the default keyword
arguments (in particular `order = 1` where applicable). Each entry reads

> `nels` · dofs · nnz · memory

where memory is for the `SparseMatrixCSC{Float64, Int64}` alone (16 bytes per stored
entry plus 8 bytes per column) — expect roughly twice that as peak memory during
assembly. Dof counts are exact; nnz and memory are extrapolated from the nnz-per-row
measured at the ~10⁶ size (exact there, within a few percent elsewhere). Dof counts
grow as `nels^d`, so other targets are easy to extrapolate.

The *bandwidth* column reports the row bandwidth — the maximum distance `|i - j|` from
the diagonal to a stored entry in row `i` — as min / median / mean / max over all rows,
measured at the ~10⁶ size. With the dof numbering resulting from `generate_grid` +
`DofHandler` the matrices are banded, with bandwidth growing like dofs^(1/2) in 2d and
dofs^(2/3) in 3d.

`convection_diffusion` is structurally identical to `heat_equation` (same row shared).

### 2d

| Problem | Cell | ~10⁴ | ~10⁵ | ~10⁶ | ~10⁷ | bandwidth at ~10⁶ |
|:---|:---|:---|:---|:---|:---|:---|
| `heat_equation` | `Triangle` | 100 · 10.2k · 71k · 1.2 MB | 320 · 103k · 720k · 12 MB | 1000 · 1.00M · 7.0M · 115 MB | 3160 · 9.99M · 70M · 1.1 GB | 2 / 1.0k / 1.0k / 2.0k |
| `heat_equation` | `Quadrilateral` | 100 · 10.2k · 92k · 1.5 MB | 320 · 103k · 926k · 15 MB | 1000 · 1.00M · 9.0M · 145 MB | 3160 · 9.99M · 90M · 1.4 GB | 2 / 1.0k / 1.0k / 2.0k |
| `linear_elasticity` | `Triangle` | 70 · 10.1k · 141k · 2.2 MB | 220 · 97.7k · 1.4M · 22 MB | 710 · 1.01M · 14M · 223 MB | 2240 · 10.0M · 140M · 2.2 GB | 4 / 1.4k / 1.4k / 2.8k |
| `linear_elasticity` | `Quadrilateral` | 70 · 10.1k · 181k · 2.8 MB | 220 · 97.7k · 1.8M · 28 MB | 710 · 1.01M · 18M · 285 MB | 2240 · 10.0M · 180M · 2.8 GB | 4 / 1.4k / 1.4k / 2.8k |
| `stokes` | `Triangle` | 33 · 10.1k · 293k · 4.6 MB | 105 · 100k · 2.9M · 45 MB | 330 · 983k · 28M · 442 MB | 1050 · 9.93M · 287M · 4.4 GB | 7 / 3.0k / 3.0k / 4.6k |
| `stokes` | `Quadrilateral` | 33 · 10.1k · 401k · 6.2 MB | 105 · 100k · 4.0M · 61 MB | 330 · 983k · 39M · 601 MB | 1050 · 9.93M · 393M · 5.9 GB | 12 / 3.0k / 3.0k / 4.6k |
| `darcy` | `Triangle` | 45 · 10.2k · 59k · 1.0 MB | 140 · 98.3k · 570k · 9.4 MB | 450 · 1.01M · 5.9M · 97 MB | 1410 · 9.94M · 58M · 955 MB | 2 / 2.2k / 1.8k / 2.7k |
| `darcy` | `Quadrilateral` | 55 · 9.2k · 70k · 1.1 MB | 180 · 97.6k · 747k · 12 MB | 580 · 1.01M · 7.7M · 126 MB | 1820 · 9.94M · 76M · 1.2 GB | 3 / 1.7k / 1.7k / 2.3k |
| `dg_heat_equation` | `Triangle` | 40 · 9.6k · 115k · 1.8 MB | 130 · 101k · 1.2M · 19 MB | 410 · 1.01M · 12M · 192 MB | 1290 · 9.98M · 120M · 1.9 GB | 3 / 2.5k / 2.5k / 2.5k |
| `dg_heat_equation` | `Quadrilateral` | 50 · 10.0k · 200k · 3.1 MB | 160 · 102k · 2.0M · 32 MB | 500 · 1.00M · 20M · 312 MB | 1580 · 9.99M · 199M · 3.1 GB | 2.0k / 2.0k / 2.0k / 2.0k |

### 3d

| Problem | Cell | ~10⁴ | ~10⁵ | ~10⁶ | ~10⁷ | bandwidth at ~10⁶ |
|:---|:---|:---|:---|:---|:---|:---|
| `heat_equation` | `Tetrahedron` | 21 · 10.6k · 157k · 2.5 MB | 45 · 97.3k · 1.4M · 23 MB | 100 · 1.03M · 15M · 240 MB | 210 · 9.39M · 139M · 2.1 GB | 4 / 10.3k / 10.3k / 20.6k |
| `heat_equation` | `Hexahedron` | 21 · 10.6k · 282k · 4.4 MB | 45 · 97.3k · 2.6M · 40 MB | 100 · 1.03M · 27M · 424 MB | 210 · 9.39M · 249M · 3.8 GB | 4 / 10.3k / 10.3k / 20.6k |
| `linear_elasticity` | `Tetrahedron` | 14 · 10.1k · 445k · 6.9 MB | 31 · 98.3k · 4.3M · 67 MB | 70 · 1.07M · 47M · 729 MB | 150 · 10.3M · 454M · 6.9 GB | 12 / 15.3k / 15.3k / 30.7k |
| `linear_elasticity` | `Hexahedron` | 14 · 10.1k · 797k · 12 MB | 31 · 98.3k · 7.7M · 119 MB | 70 · 1.07M · 85M · 1.3 GB | 150 · 10.3M · 813M · 12.2 GB | 12 / 15.3k / 15.3k / 30.7k |
| `stokes` | `Tetrahedron` | 7 · 10.6k · 1.0M · 16 MB | 15 · 93.5k · 9.0M · 137 MB | 34 · 1.03M · 99M · 1.5 GB | 75 · 10.8M · 1032M · 15.5 GB | 18 / 30.7k / 30.1k / 46.6k |
| `stokes` | `Hexahedron` | 7 · 10.6k · 2.2M · 34 MB | 15 · 93.5k · 19M · 296 MB | 34 · 1.03M · 213M · 3.2 GB | 75 · 10.8M · 2232M · 33.3 GB | 48 / 30.7k / 30.2k / 46.7k |
| `darcy` | `Tetrahedron` | 8 · 9.6k · 73k · 1.2 MB | 18 · 107k · 814k · 13 MB | 38 · 996k · 7.6M · 123 MB | 80 · 9.25M · 70M · 1.1 GB | 3 / 707 / 12.9k / 29.0k |
| `darcy` | `Hexahedron` | 13 · 9.3k · 106k · 1.7 MB | 29 · 100k · 1.1M · 18 MB | 65 · 1.11M · 13M · 201 MB | 135 · 9.90M · 113M · 1.8 GB | 4 / 17.0k / 16.8k / 21.3k |
| `dg_heat_equation` | `Tetrahedron` | 7 · 8.2k · 163k · 2.5 MB | 16 · 98.3k · 1.9M · 30 MB | 35 · 1.03M · 20M · 318 MB | 75 · 10.1M · 200M · 3.1 GB | 8 / 29.4k / 19.3k / 29.4k |
| `dg_heat_equation` | `Hexahedron` | 11 · 10.6k · 586k · 9.0 MB | 23 · 97.3k · 5.4M · 83 MB | 50 · 1.00M · 55M · 848 MB | 110 · 10.7M · 586M · 8.8 GB | 20.0k / 20.0k / 20.0k / 20.0k |

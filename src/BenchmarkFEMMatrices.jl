"""
    BenchmarkFEMMatrices

Sparse matrices from finite element discretizations (using
[Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)) of various model problems, meant
as test/benchmark matrices for linear solvers and preconditioners.

Every problem is set up on the hypercube ``[-1, 1]^d`` (``d = 2`` or ``d = 3`` depending
on the cell type) meshed with `nels` elements per side, and every setup function returns
the tuple `(K, f)`: the sparse system matrix and the right hand side vector, with
Dirichlet boundary conditions applied symmetrically with `Ferrite.apply!` (constrained
rows/columns zeroed out with a diagonal entry, keeping symmetry when present).

The problems cover different matrix classes:

| Function                             | Matrix properties                                 |
|:------------------------------------ |:------------------------------------------------- |
| [`heat_equation`](@ref)              | symmetric positive definite                       |
| [`convection_diffusion`](@ref)       | nonsymmetric                                      |
| [`linear_elasticity`](@ref)          | symmetric positive definite, vector-valued/blocked |
| [`stokes`](@ref)                     | symmetric indefinite saddle point, zero (2,2)-block |
| [`darcy`](@ref)                      | symmetric indefinite saddle point, H(div) facet dofs |
| [`dg_heat_equation`](@ref)           | symmetric positive definite, dense face-coupled blocks |
"""
module BenchmarkFEMMatrices

using Ferrite

export heat_equation, convection_diffusion, linear_elasticity, stokes, darcy,
    dg_heat_equation

include("common.jl")
include("heat.jl")
include("elasticity.jl")
include("stokes.jl")
include("darcy.jl")
include("dg_heat.jl")

end # module BenchmarkFEMMatrices

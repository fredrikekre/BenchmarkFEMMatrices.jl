using BenchmarkFEMMatrices
using Ferrite: Triangle, Quadrilateral, Tetrahedron, Hexahedron
using LinearAlgebra: Symmetric, cholesky, issuccess, norm
using SparseArrays: SparseMatrixCSC, sparse
using Test

# Symmetry up to assembly roundoff
issym(K) = norm(K - sparse(K')) ≤ 1.0e-12 * norm(K)
# Positive definiteness via sparse Cholesky
isspd(K) = issym(K) && issuccess(cholesky(Symmetric(K); check = false))

const CELLTYPES_2D = (Triangle, Quadrilateral)
const CELLTYPES_3D = (Tetrahedron, Hexahedron)

function basic_checks(K, f)
    @test K isa SparseMatrixCSC{Float64, Int}
    @test size(K, 1) == size(K, 2) == length(f)
    @test any(!iszero, f)
    u = K \ f
    @test all(isfinite, u)
    return u
end

@testset "BenchmarkFEMMatrices" begin
    @testset "heat_equation ($CT, order = $order)" for CT in (CELLTYPES_2D..., CELLTYPES_3D...), order in (1, 2)
        nels = CT in CELLTYPES_2D ? 8 : 4
        K, f = heat_equation(CT; nels = nels, order = order)
        basic_checks(K, f)
        @test isspd(K)
    end

    @testset "convection_diffusion ($CT)" for CT in (Triangle, Tetrahedron)
        nels = CT === Triangle ? 8 : 4
        K, f = convection_diffusion(CT; nels = nels)
        basic_checks(K, f)
        @test !issym(K)
        # Higher diffusivity => relatively less nonsymmetric
        K2, _ = convection_diffusion(CT; nels = nels, diffusivity = 1.0)
        asym(A) = norm(A - sparse(A')) / norm(A)
        @test asym(K2) < asym(K)
    end

    @testset "linear_elasticity ($CT, order = $order)" for CT in (CELLTYPES_2D..., CELLTYPES_3D...), order in (1, 2)
        nels = CT in CELLTYPES_2D ? 8 : 3
        K, f = linear_elasticity(CT; nels = nels, order = order)
        basic_checks(K, f)
        @test isspd(K)
        # Near-incompressible variant still assembles and is SPD
        K2, _ = linear_elasticity(CT; nels = nels, order = order, ν = 0.49)
        @test isspd(K2)
    end

    @testset "stokes ($CT)" for CT in (CELLTYPES_2D..., CELLTYPES_3D...)
        nels = CT in CELLTYPES_2D ? 8 : 3
        K, f = stokes(CT; nels = nels)
        basic_checks(K, f)
        @test issym(K)
        @test !isspd(K) # saddle point: indefinite
    end

    @testset "darcy ($CT)" for CT in (CELLTYPES_2D..., CELLTYPES_3D...)
        nels = CT in CELLTYPES_2D ? 8 : 4
        K, f = darcy(CT; nels = nels)
        basic_checks(K, f)
        @test issym(K)
        @test !isspd(K) # saddle point: indefinite
        # The barrier must not silently disappear on coarse meshes
        K2, _ = darcy(CT; nels = nels, barrier_permeability = 1.0)
        @test norm(K - K2) > 0
    end

    @testset "dg_heat_equation ($CT, order = $order)" for CT in (CELLTYPES_2D..., CELLTYPES_3D...), order in (1, 2)
        nels = CT in CELLTYPES_2D ? 8 : 4
        K, f = dg_heat_equation(CT; nels = nels, order = order)
        basic_checks(K, f)
        @test isspd(K)
    end

    @testset "nels tuple" begin
        K, f = heat_equation(Quadrilateral; nels = (4, 6))
        @test size(K, 1) == 5 * 7
        @test_throws ArgumentError heat_equation(Quadrilateral; nels = (4, 4, 4))
    end
end

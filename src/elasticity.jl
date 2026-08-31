"""
    linear_elasticity(CellType; nels = 20, order = 1, E = 1.0, ν = 0.3) -> (K, f)

Linear elasticity on ``[-1, 1]^d``: a cantilever clamped on the left boundary
(``u = 0``), loaded by a unit body force in the negative ``x_d`` direction, with an
isotropic linear elastic material (Young's modulus `E`, Poisson's ratio `ν`; plane strain
in 2d). Discretized with continuous vector-valued Lagrange elements of order `order`.

The matrix is symmetric positive definite with `dim` dofs per node. Increasing `ν`
towards `0.5` makes the problem increasingly ill-conditioned.
"""
function linear_elasticity(
        CT::Type{<:Ferrite.AbstractCell}; nels = 20, order::Int = 1,
        E::Float64 = 1.0, ν::Float64 = 0.3
    )
    grid = setup_grid(CT, nels)
    RS = Ferrite.getrefshape(CT)
    dim = Ferrite.getrefdim(CT)
    ip = Lagrange{RS, order}()^dim
    qr = QuadratureRule{RS}(2 * order)
    cv = CellValues(qr, ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), Returns(zero(Vec{dim}))))
    close!(ch)
    # Elastic stiffness C = λ I ⊗ I + 2μ Iˢʸᵐ
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    δ(i, j) = i == j ? 1.0 : 0.0
    C = SymmetricTensor{4, dim}(
        (i, j, k, l) -> λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))
    )
    b = Vec{dim}(i -> i == dim ? -1.0 : 0.0) # body force
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    fe = zeros(n)
    εs = [zero(SymmetricTensor{2, dim}) for _ in 1:n]
    σs = [zero(SymmetricTensor{2, dim}) for _ in 1:n]
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        fill!(fe, 0)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for i in 1:n
                εs[i] = shape_symmetric_gradient(cv, qp, i)
                σs[i] = C ⊡ εs[i]
            end
            for i in 1:n
                δu = shape_value(cv, qp, i)
                fe[i] += (δu ⋅ b) * dΩ
                for j in 1:n
                    Ke[i, j] += (εs[i] ⊡ σs[j]) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    apply!(K, f, ch)
    return K, f
end

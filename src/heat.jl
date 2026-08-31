"""
    heat_equation(CellType; nels = 20, order = 1) -> (K, f)

The Poisson problem ``-Δu = 1`` on ``[-1, 1]^d`` with ``u = 0`` on the whole boundary,
discretized with continuous Lagrange elements of order `order`.

The matrix is symmetric positive definite.
"""
function heat_equation(CT::Type{<:Ferrite.AbstractCell}; nels = 20, order::Int = 1)
    return assemble_scalar_diffusion(CT, nels, order, 1.0, nothing)
end

"""
    convection_diffusion(CellType; nels = 20, order = 1, diffusivity = 0.01) -> (K, f)

The convection-diffusion problem ``-κΔu + w ⋅ ∇u = 1`` on ``[-1, 1]^d`` with ``u = 0`` on
the whole boundary, discretized with continuous Lagrange elements of order `order`
(standard Galerkin, no stabilization). The wind ``w = (x_2, -x_1[, 0])`` is a rotating
field, so the strength of the advection varies over the domain, and `diffusivity` is the
knob for how dominant, and thereby how nonsymmetric, the advective term is: the maximum
cell Péclet number is ``√2 h / (2κ)`` with ``h = 2 / \\mathtt{nels}``.

The matrix is nonsymmetric.
"""
function convection_diffusion(
        CT::Type{<:Ferrite.AbstractCell}; nels = 20, order::Int = 1, diffusivity::Float64 = 0.01
    )
    return assemble_scalar_diffusion(CT, nels, order, diffusivity, rotating_wind)
end

rotating_wind(x::Vec{2}) = Vec(x[2], -x[1])
rotating_wind(x::Vec{3}) = Vec(x[2], -x[1], 0.0)

function assemble_scalar_diffusion(CT, nels, order, κ, wind::W) where {W}
    grid = setup_grid(CT, nels)
    RS = Ferrite.getrefshape(CT)
    dim = Ferrite.getrefdim(CT)
    ip = Lagrange{RS, order}()
    qr = QuadratureRule{RS}(2 * order)
    cv = CellValues(qr, ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, boundary_facetset(grid, dim), Returns(0.0)))
    close!(ch)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    fe = zeros(n)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        fill!(fe, 0)
        coords = getcoordinates(cell)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            w = wind === nothing ? nothing : wind(spatial_coordinate(cv, qp, coords))
            for i in 1:n
                δu = shape_value(cv, qp, i)
                ∇δu = shape_gradient(cv, qp, i)
                fe[i] += δu * dΩ
                for j in 1:n
                    ∇u = shape_gradient(cv, qp, j)
                    Ke[i, j] += κ * (∇δu ⋅ ∇u) * dΩ
                    if w !== nothing
                        u = shape_value(cv, qp, j)
                        Ke[i, j] += δu * (w ⋅ ∇u) * dΩ
                    end
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    apply!(K, f, ch)
    return K, f
end

"""
    dg_heat_equation(CellType; nels = 20, order = 1) -> (K, f)

The Poisson problem ``-Δu = 1`` on ``[-1, 1]^d`` discretized with the symmetric interior
penalty discontinuous Galerkin method (SIPG) with discontinuous Lagrange elements of
order `order` and penalty parameter ``μ = (1 + \\mathtt{order})^d / h_e``. Dirichlet
boundary conditions ``u = ∓1`` on the left/right boundaries (enforced strongly on the
boundary dofs), Neumann conditions on the remaining boundaries.

The matrix is symmetric positive definite, but in contrast to [`heat_equation`](@ref)
the sparsity pattern consists of dense element blocks coupled across every interior
facet, with no M-matrix property.
"""
function dg_heat_equation(CT::Type{<:Ferrite.AbstractCell}; nels = 20, order::Int = 1)
    grid = setup_grid(CT, nels)
    RS = Ferrite.getrefshape(CT)
    dim = Ferrite.getrefdim(CT)
    topology = ExclusiveTopology(grid)
    ip = DiscontinuousLagrange{RS, order}()
    qr = QuadratureRule{RS}(2 * order)
    facet_qr = FacetQuadratureRule{RS}(2 * order)
    cv = CellValues(qr, ip)
    fv = FacetValues(facet_qr, ip)
    iv = InterfaceValues(facet_qr, ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"), Returns(1.0)))
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), Returns(-1.0)))
    close!(ch)
    # Cross-element coupling must be requested explicitly in the sparsity pattern.
    K = allocate_matrix(dh; interface_coupling = [true;;], topology = topology)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    n = ndofs_per_cell(dh)
    # Volume contributions
    Ke = zeros(n, n)
    fe = zeros(n)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        fill!(fe, 0)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for i in 1:n
                fe[i] += shape_value(cv, qp, i) * dΩ
                ∇δu = shape_gradient(cv, qp, i)
                for j in 1:n
                    Ke[i, j] += (∇δu ⋅ shape_gradient(cv, qp, j)) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    # Interface contributions
    Ki = zeros(2n, 2n)
    for ic in InterfaceIterator(dh, topology)
        reinit!(iv, ic)
        hₑ = diameter(∩(getcoordinates(ic)...))
        μ = (1 + order)^dim / hₑ
        fill!(Ki, 0)
        for qp in 1:getnquadpoints(iv)
            normal = getnormal(iv, qp)
            dΓ = getdetJdV(iv, qp)
            for i in 1:getnbasefunctions(iv)
                δu_jump = shape_value_jump(iv, qp, i) * (-normal)
                ∇δu_avg = shape_gradient_average(iv, qp, i)
                for j in 1:getnbasefunctions(iv)
                    u_jump = shape_value_jump(iv, qp, j) * (-normal)
                    ∇u_avg = shape_gradient_average(iv, qp, j)
                    Ki[i, j] += (-(δu_jump ⋅ ∇u_avg + ∇δu_avg ⋅ u_jump) + μ * (δu_jump ⋅ u_jump)) * dΓ
                end
            end
        end
        assemble!(assembler, interfacedofs(ic), Ki)
    end
    # Neumann boundary contributions: prescribed flux g = n ⋅ e₂
    ∂Ωₙ = boundary_facetset(grid, dim; except = ("left", "right"))
    for fc in FacetIterator(dh, ∂Ωₙ)
        reinit!(fv, fc)
        fill!(fe, 0)
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            g = getnormal(fv, qp)[2]
            for i in 1:getnbasefunctions(fv)
                fe[i] += g * shape_value(fv, qp, i) * dΓ
            end
        end
        assemble!(f, celldofs(fc), fe)
    end
    apply!(K, f, ch)
    return K, f
end

function diameter(coords::Vector{<:Vec})
    d = 0.0
    for i in eachindex(coords), j in (i + 1):lastindex(coords)
        d = max(d, norm(coords[i] - coords[j]))
    end
    return d
end

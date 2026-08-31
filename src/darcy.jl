"""
    darcy(CellType; nels = 20, barrier_permeability = 1.0e-4) -> (K, f)

Darcy flow (the mixed form of the Poisson problem) through ``[-1, 1]^d`` with an almost
impermeable barrier: the band ``|x_1| ≤ 0.2`` has permeability `barrier_permeability`
(surrounding material: 1), except for a gap around the center that the flow is forced
through. The pressure is prescribed weakly to 1 on the left and 0 on the right boundary,
and the remaining boundaries are impermeable (``q ⋅ n = 0``, enforced with
`ProjectedDirichlet`). Discretized with the lowest order Raviart-Thomas element for the
flux and element-wise constants for the pressure.

The matrix is a symmetric indefinite saddle point matrix where, in contrast to
[`stokes`](@ref), the (1,1)-block is a (permeability-weighted) mass matrix, and the flux
dofs live on element facets. Lowering `barrier_permeability` increases the material
contrast and thereby the conditioning difficulty.
"""
function darcy(CT::Type{<:Ferrite.AbstractCell}; nels = 20, barrier_permeability::Float64 = 1.0e-4)
    RS = Ferrite.getrefshape(CT)
    dim = Ferrite.getrefdim(CT)
    nels isa Integer && (nels = ntuple(_ -> Int(nels), dim))
    grid = setup_grid(CT, nels)
    # Permeability: almost impermeable barrier in the band |x₁| ≤ 0.2, with a gap where
    # all remaining coordinates satisfy |xᵢ| < 0.2. Cells are classified by their
    # centroid, and the band/gap widened to the element size if necessary, so that the
    # barrier is at least one cell layer thick (and the gap at least one cell layer wide)
    # also on coarse meshes.
    half = ntuple(i -> max(0.2, 2.0 / nels[i]), dim)
    k = fill(1.0, getncells(grid))
    for cellid in 1:getncells(grid)
        coords = getcoordinates(grid, cellid)
        x̄ = sum(coords) / length(coords)
        if abs(x̄[1]) ≤ half[1] && any(i -> abs(x̄[i]) ≥ half[i], 2:dim)
            k[cellid] = barrier_permeability
        end
    end
    ip_q = RaviartThomas{RS, 1}()
    ip_p = DiscontinuousLagrange{RS, 0}()
    ip_geo = Lagrange{RS, 1}()
    qr = QuadratureRule{RS}(2)
    cv_q = CellValues(qr, ip_q, ip_geo)
    cv_p = CellValues(qr, ip_p, ip_geo)
    facet_qr = FacetQuadratureRule{RS}(2)
    fv_q = FacetValues(facet_qr, ip_q, ip_geo)
    dh = DofHandler(grid)
    add!(dh, :q, ip_q)
    add!(dh, :p, ip_p)
    close!(dh)
    ch = ConstraintHandler(dh)
    impermeable = boundary_facetset(grid, dim; except = ("left", "right"))
    add!(ch, ProjectedDirichlet(:q, impermeable, (x, t, n) -> 0.0))
    close!(ch)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    range_q = dof_range(dh, :q)
    range_p = dof_range(dh, :p)
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(cv_q, cell)
        reinit!(cv_p, cell)
        fill!(Ke, 0)
        kᵉ = k[cellid(cell)]
        for qp in 1:getnquadpoints(cv_q)
            dΩ = getdetJdV(cv_q, qp)
            for (i, I) in pairs(range_q)
                δq = shape_value(cv_q, qp, i)
                divδq = shape_divergence(cv_q, qp, i)
                for (j, J) in pairs(range_q)
                    Ke[I, J] += (δq ⋅ shape_value(cv_q, qp, j)) / kᵉ * dΩ
                end
                for (j, J) in pairs(range_p)
                    Ke[I, J] -= divδq * shape_value(cv_p, qp, j) * dΩ
                end
            end
            for (i, I) in pairs(range_p)
                δp = shape_value(cv_p, qp, i)
                for (j, J) in pairs(range_q)
                    Ke[I, J] -= δp * shape_divergence(cv_q, qp, j) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    # The prescribed pressure p = 1 on the left boundary enters weakly through the rhs
    # (p = 0 on the right boundary contributes nothing).
    fe = zeros(n)
    for facet in FacetIterator(dh, getfacetset(grid, "left"))
        reinit!(fv_q, facet)
        fill!(fe, 0)
        for qp in 1:getnquadpoints(fv_q)
            dΓ = getdetJdV(fv_q, qp)
            n_qp = getnormal(fv_q, qp)
            for (i, I) in pairs(range_q)
                δq = shape_value(fv_q, qp, i)
                fe[I] -= (δq ⋅ n_qp) * 1.0 * dΓ
            end
        end
        assemble!(f, celldofs(facet), fe)
    end
    apply!(K, f, ch)
    return K, f
end

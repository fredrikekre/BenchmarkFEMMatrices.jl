"""
    stokes(CellType; nels = 20) -> (K, f)

Stokes flow in a lid-driven cavity on ``[-1, 1]^d``: no-slip conditions (``u = 0``) on
all walls, lid velocity ``u = (1, 0[, 0])`` on the top boundary, and the pressure pinned
to zero in one corner vertex to remove the constant pressure mode. Discretized with
Taylor-Hood elements: continuous quadratic Lagrange for the velocity and continuous
linear Lagrange for the pressure.

The matrix is a symmetric indefinite saddle point matrix with a zero (2,2)-block (no
pressure-pressure coupling in the sparsity pattern).
"""
function stokes(CT::Type{<:Ferrite.AbstractCell}; nels = 20)
    grid = setup_grid(CT, nels)
    RS = Ferrite.getrefshape(CT)
    dim = Ferrite.getrefdim(CT)
    ipu = Lagrange{RS, 2}()^dim
    ipp = Lagrange{RS, 1}()
    qr = QuadratureRule{RS}(4)
    cvu = CellValues(qr, ipu)
    cvp = CellValues(qr, ipp; update_gradients = false, update_detJdV = false)
    dh = DofHandler(grid)
    add!(dh, :u, ipu)
    add!(dh, :p, ipp)
    close!(dh)
    # Boundary conditions: no-slip walls, moving lid (the lid condition is added last and
    # thus takes precedence in the lid corners), and one pinned pressure dof.
    addvertexset!(grid, "pressure_pin", x -> all(xi -> xi ≈ -1.0, x))
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, boundary_facetset(grid, dim; except = ("top",)), Returns(zero(Vec{dim}))))
    add!(ch, Dirichlet(:u, getfacetset(grid, "top"), Returns(Vec{dim}(i -> i == 1 ? 1.0 : 0.0))))
    add!(ch, Dirichlet(:p, getvertexset(grid, "pressure_pin"), Returns(0.0)))
    close!(ch)
    K = allocate_matrix(dh, ch; coupling = [true true; true false])
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    range_u = dof_range(dh, :u)
    range_p = dof_range(dh, :p)
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    fe = zeros(n)
    for cell in CellIterator(dh)
        reinit!(cvu, cell)
        reinit!(cvp, cell)
        fill!(Ke, 0)
        fill!(fe, 0)
        for qp in 1:getnquadpoints(cvu)
            dΩ = getdetJdV(cvu, qp)
            for (i, I) in pairs(range_u)
                ∇δu = shape_gradient(cvu, qp, i)
                divδu = shape_divergence(cvu, qp, i)
                for (j, J) in pairs(range_u)
                    Ke[I, J] += (∇δu ⊡ shape_gradient(cvu, qp, j)) * dΩ
                end
                for (j, J) in pairs(range_p)
                    Ke[I, J] -= divδu * shape_value(cvp, qp, j) * dΩ
                end
            end
            for (i, I) in pairs(range_p)
                δp = shape_value(cvp, qp, i)
                for (j, J) in pairs(range_u)
                    Ke[I, J] -= δp * shape_divergence(cvu, qp, j) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    apply!(K, f, ch)
    return K, f
end

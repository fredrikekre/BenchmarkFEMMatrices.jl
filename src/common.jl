# Internal helpers shared by the problem setups.

# Mesh the domain [-1, 1]^d with nels elements per side (or nels[i] elements in
# direction i if a tuple is passed).
function setup_grid(CT::Type{<:Ferrite.AbstractCell}, nels::Union{Integer, Tuple{Vararg{Integer}}})
    dim = Ferrite.getrefdim(CT)
    nels isa Integer && (nels = ntuple(_ -> Int(nels), dim))
    length(nels) == dim ||
        throw(ArgumentError("length(nels) = $(length(nels)) does not match the grid dimension $(dim)"))
    return generate_grid(CT, nels)
end

# Facetset names used by generate_grid, ordered such that the first 2d entries exist for
# a d-dimensional grid.
const FACETSET_NAMES = ("left", "right", "top", "bottom", "front", "back")

boundary_facetsets(grid, dim) = (getfacetset(grid, name) for name in FACETSET_NAMES[1:(2 * dim)])

# The union of all boundary facets, optionally excluding some facetsets by name.
function boundary_facetset(grid, dim; except::Tuple{Vararg{String}} = ())
    names = filter(!in(except), FACETSET_NAMES[1:(2 * dim)])
    return union((getfacetset(grid, name) for name in names)...)
end

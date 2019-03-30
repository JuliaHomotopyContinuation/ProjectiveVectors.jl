using Documenter, ProjectiveVectors
using LinearAlgebra

makedocs(
    format = Documenter.HTML(),
    sitename = "ProjectiveVectors",
    pages = [
        "Index" => "index.md"
        ],
    doctest=false,
    modules=[ProjectiveVectors],
    checkdocs=:exports
)

deploydocs(
    repo   = "github.com/JuliaHomotopyContinuation/ProjectiveVectors.jl.git",
    target = "build",
    julia = "1.1",
    osname = "linux",
    deps   = nothing,
    make   = nothing
)

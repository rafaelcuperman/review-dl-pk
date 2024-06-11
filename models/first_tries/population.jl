struct Individual
    id::Integer
    x::Vector{Float64}
    t::Vector{Float64}
    y::Vector{Float64}
    cols_x::Vector{String}
    callback
    function Individual(; id, x, t, y=zeros(length(t)), cols_x=["WEIGHT", "AGE", "SEX", "AMT"], callback = nothing) 
        length(t) == length(y) ? new(id,x,t,y,cols_x,callback) : error("Lengths of vectors t and y don't match")
    end
    
end

struct Population
    individuals::Vector{Individual}
    function Population(individuals) #Check that there is a unique individual per ID
        ids = []
        for i in individuals
            push!(ids, i.id)
        end
        return length(ids) == length(Set(ids)) ? new(individuals) : error("Repeated individual ids")
    end
end

function get_individual_by_id(p::Population, id::Int)
    for ind in p.individuals
        if ind.id == id
            return ind
        end
    end
end

# Concatenate time and covariates. Builds X matrix for DNN for an individual. Time is the first column
function create_X(i::Individual)  
    return hcat(i.t, repeat(i.x',length(i.t),1))
end

function create_X(i::Vector{Individual})
    return reduce(vcat, create_X.(i))
end

function create_X(p::Population)
    return reduce(vcat, create_X.(p.individuals))
end

function create_y(i::Individual)  
    return i.y
end

function create_y(i::Vector{Individual})  
    return reduce(vcat, create_y.(i))
end

function create_y(p::Population)  
    return reduce(vcat, create_y.(p.individuals))
end

function create_X_y(i::Individual)
    return create_X(i), create_y(i)
end

function create_X_y(i::Vector{Individual})
    return create_X(i), create_y(i)
end

function create_X_y(p::Population)
    return create_X(p), create_y(p)
end

Base.getindex(pop::Population, i::Int) = pop.individuals[i]
Base.getindex(pop::Population, i::Vector{Int64}) = pop.individuals[i]
Base.length(pop::Population) = length(pop.individuals)

function get_property(pop::Population, f::Symbol)
    return getfield.(pop.individuals, f)
end

function get_property(i::Vector{Individual}, f::Symbol)
    return getfield.(i, f)
end

function get_property(i::Individual, f::Symbol)
    return getfield(i, f)
end

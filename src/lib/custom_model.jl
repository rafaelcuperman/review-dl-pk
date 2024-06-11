


struct CustomLuxModel{O,M,P} <: AbstractModel{O,M,P}
    objective::O
    model::M
    p::P
    kwargs
end 
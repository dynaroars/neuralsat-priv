import gurobipy as gp
from gurobipy import GRB

# Create a new model
model = gp.Model("initial_model")

# Add variables with bounds
x1 = model.addVar(lb=0, ub=10, name="x1")
x2 = model.addVar(lb=0, ub=15, name="x2")
x3 = model.addVar(lb=0, ub=12, name="x3")
x4 = model.addVar(lb=0, ub=8, name="x4")
x5 = model.addVar(lb=0, ub=20, name="x5")

# Add complex linear constraints
model.addConstr(3*x1 + 2*x2 + x3 + x4 + 5*x5 <= 50, name="c1")
model.addConstr(2*x1 - x2 + 3*x3 - 2*x4 + x5 >= 15, name="c2")
model.addConstr(-x1 + 4*x2 - x3 + x4 + 3*x5 == 20, name="c3")
model.addConstr(2*x1 + 3*x2 - x4 <= 30, name="c4")
model.addConstr(x1 + 2*x3 + x5 >= 10, name="c5")
model.addConstr(4*x1 - x2 + x3 + 3*x4 - x5 <= 25, name="c6")
model.addConstr(3*x1 + x2 + 4*x3 + 2*x5 >= 18, name="c7")
model.addConstr(-2*x1 + x4 + 2*x5 <= 12, name="c8")
model.addConstr(5*x1 + x3 + 2*x4 - x5 == 35, name="c9")
model.addConstr(3*x1 + 2*x2 + x3 - x4 + x5 >= 22, name="c10")

# Set a more complex objective function
model.setObjective(4*x1 + 3*x2 + 5*x3 + 2*x4 + x5, GRB.MAXIMIZE)

model.update()
print(model)

# Optimize the model
model.optimize()

# Print results
print("Initial model results:")
if model.status == GRB.OPTIMAL:
    for v in model.getVars():
        print(f'{v.varName}: {v.x}')
    print(f'Objective: {model.objVal}')

# exit()

print('\n\n################ new model ################\n\n')
# Clone the model
new_model = model.copy()

# Remove the constraint 'c3'
constraint_to_remove = new_model.getConstrByName('c10')
print(constraint_to_remove)
new_model.remove(constraint_to_remove)

# Update the model to finalize the removal of the constraint
new_model.update()

# Re-optimize the model
new_model.optimize()

# Print results
print("Modified model results (after removing c3):")
if new_model.status == GRB.OPTIMAL:
    for v in new_model.getVars():
        print(f'{v.varName}: {v.x}')
    print(f'Objective: {new_model.objVal}')

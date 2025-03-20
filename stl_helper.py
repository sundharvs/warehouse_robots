import gurobipy as gp
from gurobipy import GRB



# Helper functions to add STL constraints
def stl_eventually(model,start, end, var, op, threshold):
    """Add constraints to enforce that var[t] op threshold is true for at least one t in [start, end]. 
       Returns a binary indicator for the satisfaction of the eventuality."""
    # Create binary indicators for each time in [start, end]
    b = model.addVars(range(start, end+1), vtype=GRB.BINARY, name=f"b_event_{start}_{end}")
    # Link b[t] to the condition var[t] op threshold
    M = 1e6  # a sufficiently large constant (Big M method)
    for t in range(start, end+1):
        if op == ">=":
            model.addConstr(var[t] >= threshold - M * (1 - b[t]), name=f"event_ge_low_{t}")
            model.addConstr(var[t] <= (threshold - 1e-6) + M * b[t], name=f"event_ge_high_{t}")
        elif op == "<=":
            model.addConstr(var[t] <= threshold + M * (1 - b[t]), name=f"event_le_high_{t}")
            model.addConstr(var[t] >= (threshold + 1e-6) - M * b[t], name=f"event_le_low_{t}")
        else:
            raise ValueError("Unsupported operator")
    # One of the b[t] must be 1 (at least one time satisfies the condition)
    phi = model.addVar(vtype=GRB.BINARY, name=f"phi_event_{start}_{end}")
    model.addConstr(gp.quicksum(b[t] for t in range(start, end+1)) >= 1 * phi, name=f"event_any_{start}_{end}")
    model.addConstr(phi <= gp.quicksum(b[t] for t in range(start, end+1)), name=f"event_any2_{start}_{end}")
    return phi  # return binary var representing the truth of the "eventually" formula

def stl_globally(model,start, end, var, op, threshold):
    """Add constraints to enforce that var[t] op threshold for all t in [start, end].
       Returns a binary indicator for the satisfaction of the global condition."""
    phi = model.addVar(vtype=GRB.BINARY, name=f"phi_global_{start}_{end}")
    M = 1e6
    for t in range(start, end+1):
        if op == ">=":
            # If phi is true, enforce var[t] >= threshold; if phi is false, no restriction (allow failing)
            model.addConstr(var[t] >= threshold - M * (1 - phi), name=f"glob_ge_{t}")
        elif op == "<=":
            model.addConstr(var[t] <= threshold + M * (1 - phi), name=f"glob_le_{t}")
        else:
            raise ValueError("Unsupported operator")
    # If phi=1, all above constraints force condition to hold for every t in [start,end].
    # If any constraint would violate, the solver will set phi=0 to satisfy feasibility.
    return phi

def stl_and(model,phi_a, phi_b):
    """Return a binary that is 1 iff both phi_a and phi_b are 1."""
    phi = model.addVar(vtype=GRB.BINARY, name="phi_and")
    # phi <= each of (phi_a, phi_b)
    model.addConstr(phi <= phi_a, name="and_le_a")
    model.addConstr(phi <= phi_b, name="and_le_b")
    # phi >= phi_a + phi_b - 1   (phi = 1 only if both are 1)
    model.addConstr(phi >= phi_a + phi_b - 1, name="and_req_both")
    return phi

def stl_or(model,phi_a, phi_b):
    """Return a binary that is 1 iff either phi_a or phi_b is 1."""
    phi = model.addVar(vtype=GRB.BINARY, name="phi_or")
    # phi >= each of (phi_a, phi_b)
    model.addConstr(phi >= phi_a, name="or_ge_a")
    model.addConstr(phi >= phi_b, name="or_ge_b")
    # phi <= phi_a + phi_b   (phi = 0 only if both are 0)
    model.addConstr(phi <= phi_a + phi_b, name="or_req_one")
    return phi

def stl_not(model, phi, name):
    """
    Return a binary variable representing the logical negation of phi.

    Args:
        model (gurobipy.Model): The Gurobi optimization model.
        phi (gurobipy.Var): The binary variable to negate.
        name (str): A name for the new negation variable.

    Returns:
        gurobipy.Var: Binary variable indicating NOT(phi).
    """
    phi_not = model.addVar(vtype=GRB.BINARY, name=name)
    # phi_not = 1 - phi
    model.addConstr(phi + phi_not == 1, name=f"{name}_negation")
    return phi_not
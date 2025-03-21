from tracemalloc import start
import gurobipy as gp
from gurobipy import GRB

from GridWorld import GridWorld
import random


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


def get_incremental_cost(original_completion_time,updated_completion_time,t_h,alpha=1,beta=1):
    """Computes
      (1) t_h (time it took for the requester to be helped), i.e time to help_site
       (2) t*, the additional time it cost for the helper
       (3) cost = alpha* t_h + betha * t* where default value is alpha =1 and beta =1 
       """
    #first compute t_original
    t_original = original_completion_time
    #Then compute t_updated
    t_updated = updated_completion_time
    t_star = t_updated-t_original
    t_h = t_h
    cost = alpha*t_star + beta*t_h
    return t_star, t_h, cost


def count_transitions_until_target(path, target_location):
    if len(path) < 2:
        return 0

    transitions = 0
    prev_point = path[0]
    #need to add the case where we start at helpsite
    if path[0] == target_location:
        return 0
        

    for point in path[1:]:
        if point != prev_point:
            transitions += 1
        if point == target_location:
            break
        prev_point = point

    return transitions


def start_pos_init(grid:GridWorld,needs_help=False):
    if needs_help:
        start_pos = grid.conflict_cell
    else:
        free_except_conflict = [c for c in grid.free_cells if c != grid.conflict_cell]
        start_pos = random.choice(free_except_conflict)
    return start_pos

def get_ltl_spec(grid: GridWorld, num_existing_locs=2, needs_help=False):
    ltl_spec = []
    if needs_help:
        ltl_spec.append(grid.conflict_cell)
        
        interest_points = random.sample(
            [c for c in grid.free_cells if c != grid.dynamic_conflict and c != grid.conflict_cell],
            num_existing_locs - 1
        )
        ltl_spec.extend(interest_points)
    else:
        interest_points =random.sample(
        [c for c in grid.free_cells if c != grid.dynamic_conflict], num_existing_locs)
        ltl_spec.extend(interest_points)
        
    return ltl_spec



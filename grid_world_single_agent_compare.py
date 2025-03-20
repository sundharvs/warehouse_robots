from re import U
from turtle import update
from openai import OpenAI
import random
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from gridworld import GridWorld
from stl_helper import stl_and, stl_eventually, stl_globally, stl_or
client = OpenAI()
class GridWorldAgent:

    def __init__(self, name,grid:GridWorld,start_pos,interest_points,T,actuation=1):
        self
        self.name=name
        self.grid = grid
        self.start_pos=start_pos
        self.interest_points = interest_points
        self.T=T
        self.actuation=actuation

        self.model = gp.Model(f"STL_gridworld_{self.name}")

        # Will hold the Gurobi variables
        self.b = {}     # Position indicator b[t, i, j]
        self.x = {}     # x[t]
        self.y = {}     # y[t]
        self.dx = {}    # dx[t]
        self.dy = {}    # dy[t]

        #initiate the model
        self._build_model()

    def _build_model(self):
        """Internal method to create Gurobi variables, constraints, objective."""
        # ---------------
        # 2.1. Variables
        # ---------------
        T = self.T
        free_cells = self.grid.free_cells
        grid_size = self.grid.grid_size
        model = self.model

        # Position indicator variables
        for t in range(T+1):
            for (i, j) in free_cells:
                self.b[t, i, j] = model.addVar(
                    vtype=GRB.BINARY, name=f"b_{self.name}_{t}_{i}_{j}"
                )

        # Coordinates (integer)
        for t in range(T+1):
            self.x[t] = model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=grid_size-1,
                name=f"x_{self.name}_{t}"
            )
            self.y[t] = model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=grid_size-1,
                name=f"y_{self.name}_{t}"
            )

        # Movement deltas (integer, but can range negative to positive)
        for t in range(T):
            self.dx[t] = model.addVar(
                vtype=GRB.INTEGER, lb=-grid_size, ub=grid_size,
                name=f"dx_{self.name}_{t}"
            )
            self.dy[t] = model.addVar(
                vtype=GRB.INTEGER, lb=-grid_size, ub=grid_size,
                name=f"dy_{self.name}_{t}"
            )

        model.update()

        # ---------------
        # 2.2. Constraints
        # ---------------
        # Initial position
        (start_i, start_j) = self.start_pos
        model.addConstr(self.x[0] == start_i, name=f"start_x_{self.name}")
        model.addConstr(self.y[0] == start_j, name=f"start_y_{self.name}")
        # The binary indicator for the start position
        model.addConstr(self.b[0, start_i, start_j] == 1, 
                        name=f"start_b_{self.name}")

        # Exactly one position at each time t
        for t in range(T+1):
            model.addConstr(
                gp.quicksum(self.b[t, i, j] for (i, j) in free_cells) == 1,
                name=f"one_pos_{self.name}_t{t}"
            )
            # Link x[t], y[t] to selected b[t,i,j]
            model.addConstr(
                self.x[t] == gp.quicksum(i * self.b[t, i, j] 
                                         for (i, j) in free_cells),
                name=f"link_x_{self.name}_t{t}"
            )
            model.addConstr(
                self.y[t] == gp.quicksum(j * self.b[t, i, j] 
                                         for (i, j) in free_cells),
                name=f"link_y_{self.name}_t{t}"
            )
        #also add a notion of time it takes

        self.all_points_visited = {}
        for t in range(self.T + 1):
            self.all_points_visited[t] = self.model.addVar(vtype=GRB.BINARY, name=f"all_points_visited_{self.name}_{t}")

        M = len(self.interest_points)  # Number of interest points

        for t in range(self.T + 1):
            # For each interest point, ensure it has been visited by timestep t at least once
            for idx, (ii, jj) in enumerate(self.interest_points):
                self.model.addConstr(
                    gp.quicksum(self.b[tau, ii, jj] for tau in range(t + 1)) >= self.all_points_visited[t],
                    name=f"{self.name}_visited_by_t{t}_point{idx}"
                )

        # Ensure once points are visited, they remain visited at subsequent steps
        for t in range(self.T):
            self.model.addConstr(self.all_points_visited[t] <= self.all_points_visited[t + 1],
                                name=f"{self.name}_visited_monotonic_t{t}")

        self.completion_time = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=self.T, name=f"completion_time_{self.name}")

        # completion_time is the earliest timestep when all points are visited
        self.model.addConstr(
            self.completion_time == gp.quicksum((1 - self.all_points_visited[t]) for t in range(self.T + 1)),
            name=f"{self.name}_completion_time_def"
)

        # Movement constraints -- with Diagnonal movement allowed

        # for t in range(T):
        #     model.addConstr(self.dx[t] == self.x[t+1] - self.x[t],
        #                     name=f"dx_def_{self.name}_t{t}")
        #     model.addConstr(self.dy[t] == self.y[t+1] - self.y[t],
        #                     name=f"dy_def_{self.name}_t{t}")

        #     # Actuation (movement) limit
        #     model.addConstr(self.dx[t] <= self.actuation, 
        #                     name=f"dx_ub_{self.name}_t{t}")
        #     model.addConstr(self.dx[t] >= -self.actuation,
        #                     name=f"dx_lb_{self.name}_t{t}")
        #     model.addConstr(self.dy[t] <= self.actuation,
        #                     name=f"dy_ub_{self.name}_t{t}")
        #     model.addConstr(self.dy[t] >= -self.actuation,
        #                     name=f"dy_lb_{self.name}_t{t}")

        for t in range(T):
            model.addConstr(self.dx[t] == self.x[t+1] - self.x[t],
                            name=f"dx_def_{self.name}_t{t}")
            model.addConstr(self.dy[t] == self.y[t+1] - self.y[t],
                            name=f"dy_def_{self.name}_t{t}")

            # Actuation (movement) limit
            model.addConstr(self.dx[t] <= self.actuation, 
                            name=f"dx_ub_{self.name}_t{t}")
            model.addConstr(self.dx[t] >= -self.actuation,
                            name=f"dx_lb_{self.name}_t{t}")
            model.addConstr(self.dy[t] <= self.actuation,
                            name=f"dy_ub_{self.name}_t{t}")
            model.addConstr(self.dy[t] >= -self.actuation,
                            name=f"dy_lb_{self.name}_t{t}")

            # Cardinal direction constraint (no diagonal)
            is_vertical_move = model.addVar(vtype=GRB.BINARY, name=f"is_vertical_{self.name}_t{t}")
            M = self.grid.grid_size

            model.addConstr(self.dx[t] <= (1 - is_vertical_move) * M, name=f"dx_zero_if_vert_pos_{self.name}_t{t}")
            model.addConstr(self.dx[t] >= -(1 - is_vertical_move) * M, name=f"dx_zero_if_vert_neg_{self.name}_t{t}")
            model.addConstr(self.dy[t] <= is_vertical_move * M, name=f"dy_zero_if_hor_pos_{self.name}_t{t}")
            model.addConstr(self.dy[t] >= -is_vertical_move * M, name=f"dy_zero_if_hor_neg_{self.name}_t{t}")
        # Eventually visit all interest points
        # sum over all time steps of b[t, i_interest, j_interest] >= 1
        for idx, (ii, jj) in enumerate(self.interest_points):
            model.addConstr(
                gp.quicksum(self.b[t, ii, jj] for t in range(T+1)) >= 1,
                name=f"{self.name}_visits_interest{idx}"
            )

        # ---------------
        # 2.3. Objective
        # ---------------
        obj = gp.QuadExpr()
        for t in range(T):
            obj += self.dx[t]*self.dx[t] + self.dy[t]*self.dy[t]
        obj += self.completion_time
        model.setObjective(obj, GRB.MINIMIZE)
        # self.model.setObjective(self.completion_time, GRB.MINIMIZE)

        model.update()

    def solve(self):
        """Solve the agent's model."""
        # Optional: set parameters
        self.model.Params.OutputFlag = 0  # turn off solver logs
        self.model.optimize()

    # ---------------
    # 2.5. Extract solution
    # ---------------
    def get_path(self):
        """
        Return a list of (i,j) for times t=0..T if solution is found.
        Otherwise returns an empty list.
        """
        if self.model.Status == GRB.OPTIMAL:
            path = []
            for t in range(self.T+1):
                # find the cell (i,j) where b[t,i,j] ~ 1
                for (i, j) in self.grid.free_cells:
                    if self.b[t, i, j].X > 0.5:
                        path.append((i, j))
                        break
            return path
        else:
            return []
        
##TO DO: Need to implement nl_to_stlpy equivalent here. Need to implement and / or , eventually and until. 
def compare_and_plot(T,grid:GridWorld,agent_name='A',num_existing_loc=2,actuation=1):
    free_except_conflict = [c for c in grid.free_cells if c != grid.conflict_cell]
    start_pos = random.choice(free_except_conflict)
    
    start_pos = random.choice(grid.free_cells)
    interest_points =random.sample(
    [c for c in grid.free_cells if c != grid.dynamic_conflict and c !=start_pos], num_existing_loc)
    grid_plot = [[0 for _ in range(grid.grid_size)] for _ in range(grid.grid_size)]
    for (i,j) in grid.obstacle_cells:
            grid_plot[i][j]=1
    (k, l) = grid.conflict_cell
    grid_plot[k][l]=2
    # print(interest_points)
    # print(grid.conflict_cell)
    updated_interested_points = interest_points.copy()
    updated_interested_points.append(grid.conflict_cell)

    
    # print(updated_interested_points)
    for (i,j) in  interest_points:
        grid_plot[i][j]=3

    agent_original = GridWorldAgent(
        name = agent_name,
        grid=grid,
        start_pos = start_pos,
        interest_points=interest_points,
        T=T,
        actuation=actuation
    )
    agent_updated = GridWorldAgent(
        name = agent_name,
        grid=grid,
        start_pos = start_pos,
        interest_points=updated_interested_points,
        T=T,
        actuation=actuation
    )
    agent_original.solve()
    agent_updated.solve()
    path_original= agent_original.get_path()
    path_updated= agent_updated.get_path()
    t_h = count_transitions_until_target(path_updated,target_location=grid.conflict_cell)


    print(f"Agent {agent_name} path:", path_original)
    print(f"(Updated Agent {agent_name} path:", path_updated)
    cmap = colors.ListedColormap(['white', 'gray', 'yellow', 'orange'])
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(grid_plot, cmap=cmap, origin='lower')

    if agent_original.model.Status == GRB.OPTIMAL:
        O_xs = [p[0] for p in path_original]
        O_ys = [p[1] for p in path_original]
        ax.plot(O_ys, O_xs, marker='o', color='red', label=f'Agent {agent_name} Path')
        ax.plot(start_pos[1], start_pos[0], marker='*', color='red', markersize=15)
        original_completion_time = int(agent_original.completion_time.X)

    if agent_updated.model.Status == GRB.OPTIMAL:
        U_xs = [p[0] for p in path_updated]
        U_ys = [p[1] for p in path_updated]
        ax.plot(U_ys, U_xs, marker='o', color='blue', label=f'Updated Agent {agent_name} Path')
        ax.plot(start_pos[1], start_pos[0], marker='*', color='blue', markersize=15)
        updated_completion_time = int(agent_updated.completion_time.X) 
    for idx, (i, j) in enumerate(interest_points, start=1):
        ax.text(j, i, f'L{idx}', color='black', ha='center', va='center')
    
    if agent_updated.model.Status == GRB.OPTIMAL and agent_original.model.Status== GRB.OPTIMAL:
        t_star, t_h , cost = get_incremental_cost(original_completion_time,updated_completion_time,t_h=t_h)
    else:
        t_star, t_h, cost = 'NA', 'NA', 'Not Applicable, help not feasible'

    ax.set_xticks(range(grid.grid_size))
    ax.set_yticks(range(grid.grid_size))
    ax.set_xticklabels(range(grid.grid_size))
    ax.set_yticklabels(range(grid.grid_size))
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.text(-1.5,-1.5,f"t_star={t_star},  t_h={t_h},  cost = {cost}",fontsize=12)
    plt.title("Gridworld Paths for Agents (Separate Models)")
    plt.show()
   


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


grid = GridWorld(grid_size=8,num_chunks=2,seed=31,num_dynamic_const=1)
print (f'LIST OF OBSTACLES{grid.obstacle_cells}')

compare_and_plot(15,grid,"A",2,1)

compare_and_plot(15,grid,'B',2,1)

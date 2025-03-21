from GridWorld import GridWorld
from Message import Message
from openai import OpenAI
import logging
from re import U
from turtle import update
from openai import OpenAI
import random
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

class GridWorldAgent:
    def __init__(self, normal_ltl_spec, capabilities, initial_pos, gridworld:GridWorld,T,actuation=1, needs_help=False):
        self.ltl_spec = normal_ltl_spec 
        self.capabilities = capabilities
        self.position = initial_pos #start_pos
        self.gridworld = gridworld
        self.id = self.gridworld.register_robot(self) #name
        self.T = T
        self.actuation=actuation
        self.needs_help = needs_help
        self.model = gp.Model(f"STL_gridworld_{self.id}")

         # Will hold the Gurobi variables
        self.b = {}     # Position indicator b[t, i, j]
        self.x = {}     # x[t]
        self.y = {}     # y[t]
        self.dx = {}    # dx[t]
        self.dy = {}    # dy[t]

        #initiate the model
        self._build_model()

    def generate_help_request(self, scene_description):
        """Query the LLM to generate a help request based on current scene description and self.capabilities"""
        
        prompt = f"""
        You are a helpful assistant, onboard a warehouse robot with capabilities which is listed on the Inferred capabilities list. 
        "Scene description" explains potential issues your robot might face. This depends on what your capabillities are. 
        Generate a clear help request to other robots if you can not resolve this issue on your own.
        However, generate a line "no_help_needed=True" if you do not need help. If you do need help, generate a line "no_help_needed=False".

        Use this thought process below
        i) Why can't I resolve this issue on my own, given my capabilities
        ii) Then what capabilties are needed to resolve this issue
        iii) Generate a clear help request by reasoning about what information other robots would need to determine if they can help. The minimum information you should include are 1) What the issue is 2) And what capabilities are required to be helpful 

        Inferred capabilities: {self.capabilities}
        Scene description: {scene_description}
        """
        
        response = self.gridworld.llm_query(prompt)
        
        help_needed = response.split("no_help_needed=")[1].split()[0] == 'False'
        if help_needed:
            help_request = response.split("Help Request")[1]
            self.gridworld.broadcast_message(self.id, Message("help_request", help_request))
            self.needs_help = True
        
    def propose_help(self, help_request, sender_id):
        """Query the LLM to propose help based on help request and self.capabilities"""
        prompt = f"""
        You are a helpful assistant, onboard a warehouse robot with capabilities which is listed on the Inferred capabilities list. 
        "Help request" contains descripton of the issue and the suggested capabilities that can help. 
        You need to evaulate if you can help by reasoning over your set of capabilities and descritption of the required task.
        Generate a clear assistance proposal to the robot if you can help resolve this issue.
        However, if you can not help, generate a line "can't_help_you:True". Also include reasoning on why you can not help. 

        The minimum information you should include are 1) What capability you have that can help 2) And some description of level of effort by analyzing "considerations" list. 

        "Help request": 
        {help_request}
        
        "Inferred capabiltiies":{self.capabilities}
        """
        
        # TODO: "Considerations": {Time to target: "5 minutes", Time to complete task: "1 minute", "Artifical cost": 1 Unit per minute"}
        
        response = self.gridworld.llm_query(prompt)
        
        return Message("help_proposal", response)
    
    def receive_message(self, sender_id, message):
        """Handle received message."""
        print(f"Robot {self.id} received from {sender_id}: {message}")
        
        if message.type == "help_request":
            help_proposal = self.propose_help(message.content, sender_id)
            self.gridworld.send_message(self.id, sender_id, help_proposal)
        elif message.type == "help_proposal":
            #TODO: Do some stuff here to evaluate proposals
            pass


    def _build_model(self):
        """Internal method to create Gurobi variables, constraints, objective."""
        # ---------------
        # 2.1. Variables
        # ---------------
        T = self.T
        free_cells = self.gridworld.free_cells
        grid_size = self.gridworld.grid_size
        model = self.model

        # Position indicator variables
        for t in range(T+1):
            for (i, j) in free_cells:
                self.b[t, i, j] = model.addVar(
                    vtype=GRB.BINARY, name=f"b_{self.id}_{t}_{i}_{j}"
                )

        # Coordinates (integer)
        for t in range(T+1):
            self.x[t] = model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=grid_size-1,
                name=f"x_{self.id}_{t}"
            )
            self.y[t] = model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=grid_size-1,
                name=f"y_{self.id}_{t}"
            )

        # Movement deltas (integer, but can range negative to positive)
        for t in range(T):
            self.dx[t] = model.addVar(
                vtype=GRB.INTEGER, lb=-grid_size, ub=grid_size,
                name=f"dx_{self.id}_{t}"
            )
            self.dy[t] = model.addVar(
                vtype=GRB.INTEGER, lb=-grid_size, ub=grid_size,
                name=f"dy_{self.id}_{t}"
            )

        model.update()

        # ---------------
        # 2.2. Constraints
        # ---------------
        # Initial position
        (start_i, start_j) = self.position
        model.addConstr(self.x[0] == start_i, name=f"start_x_{self.id}")
        model.addConstr(self.y[0] == start_j, name=f"start_y_{self.id}")
        # The binary indicator for the start position
        model.addConstr(self.b[0, start_i, start_j] == 1, 
                        name=f"start_b_{self.id}")

        # Exactly one position at each time t
        for t in range(T+1):
            model.addConstr(
                gp.quicksum(self.b[t, i, j] for (i, j) in free_cells) == 1,
                name=f"one_pos_{self.id}_t{t}"
            )
            # Link x[t], y[t] to selected b[t,i,j]
            model.addConstr(
                self.x[t] == gp.quicksum(i * self.b[t, i, j] 
                                         for (i, j) in free_cells),
                name=f"link_x_{self.id}_t{t}"
            )
            model.addConstr(
                self.y[t] == gp.quicksum(j * self.b[t, i, j] 
                                         for (i, j) in free_cells),
                name=f"link_y_{self.id}_t{t}"
            )
        #also add a notion of time it takes

        self.all_points_visited = {}
        for t in range(self.T + 1):
            self.all_points_visited[t] = self.model.addVar(vtype=GRB.BINARY, name=f"all_points_visited_{self.id}_{t}")

        M = len(self.ltl_spec)  # Number of interest points

        for t in range(self.T + 1):
            # For each interest point, ensure it has been visited by timestep t at least once
            for idx, (ii, jj) in enumerate(self.ltl_spec):
                self.model.addConstr(
                    gp.quicksum(self.b[tau, ii, jj] for tau in range(t + 1)) >= self.all_points_visited[t],
                    name=f"{self.id}_visited_by_t{t}_point{idx}"
                )

        # Ensure once points are visited, they remain visited at subsequent steps
        for t in range(self.T):
            self.model.addConstr(self.all_points_visited[t] <= self.all_points_visited[t + 1],
                                name=f"{self.id}_visited_monotonic_t{t}")

        self.completion_time = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=self.T, name=f"completion_time_{self.id}")

        # completion_time is the earliest timestep when all points are visited
        self.model.addConstr(
            self.completion_time == gp.quicksum((1 - self.all_points_visited[t]) for t in range(self.T + 1)),
            name=f"{self.id}_completion_time_def"
)

        # Movement constraints -- with Diagnonal movement allowed

        # for t in range(T):
        #     model.addConstr(self.dx[t] == self.x[t+1] - self.x[t],
        #                     name=f"dx_def_{self.id}_t{t}")
        #     model.addConstr(self.dy[t] == self.y[t+1] - self.y[t],
        #                     name=f"dy_def_{self.id}_t{t}")

        #     # Actuation (movement) limit
        #     model.addConstr(self.dx[t] <= self.actuation, 
        #                     name=f"dx_ub_{self.id}_t{t}")
        #     model.addConstr(self.dx[t] >= -self.actuation,
        #                     name=f"dx_lb_{self.id}_t{t}")
        #     model.addConstr(self.dy[t] <= self.actuation,
        #                     name=f"dy_ub_{self.id}_t{t}")
        #     model.addConstr(self.dy[t] >= -self.actuation,
        #                     name=f"dy_lb_{self.id}_t{t}")

        for t in range(T):
            model.addConstr(self.dx[t] == self.x[t+1] - self.x[t],
                            name=f"dx_def_{self.id}_t{t}")
            model.addConstr(self.dy[t] == self.y[t+1] - self.y[t],
                            name=f"dy_def_{self.id}_t{t}")

            # Actuation (movement) limit
            model.addConstr(self.dx[t] <= self.actuation, 
                            name=f"dx_ub_{self.id}_t{t}")
            model.addConstr(self.dx[t] >= -self.actuation,
                            name=f"dx_lb_{self.id}_t{t}")
            model.addConstr(self.dy[t] <= self.actuation,
                            name=f"dy_ub_{self.id}_t{t}")
            model.addConstr(self.dy[t] >= -self.actuation,
                            name=f"dy_lb_{self.id}_t{t}")

            # Cardinal direction constraint (no diagonal)
            is_vertical_move = model.addVar(vtype=GRB.BINARY, name=f"is_vertical_{self.id}_t{t}")
            M = self.gridworld.grid_size

            model.addConstr(self.dx[t] <= (1 - is_vertical_move) * M, name=f"dx_zero_if_vert_pos_{self.id}_t{t}")
            model.addConstr(self.dx[t] >= -(1 - is_vertical_move) * M, name=f"dx_zero_if_vert_neg_{self.id}_t{t}")
            model.addConstr(self.dy[t] <= is_vertical_move * M, name=f"dy_zero_if_hor_pos_{self.id}_t{t}")
            model.addConstr(self.dy[t] >= -is_vertical_move * M, name=f"dy_zero_if_hor_neg_{self.id}_t{t}")
        # Eventually visit all interest points
        # sum over all time steps of b[t, i_interest, j_interest] >= 1
        for idx, (ii, jj) in enumerate(self.ltl_spec):
            model.addConstr(
                gp.quicksum(self.b[t, ii, jj] for t in range(T+1)) >= 1,
                name=f"{self.id}_visits_interest{idx}"
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
                for (i, j) in self.gridworld.free_cells:
                    if self.b[t, i, j].X > 0.5:
                        path.append((i, j))
                        break
            return path
        else:
            return []    
        

from re import U
from turtle import position, update
from Robot import GridWorldAgent
from openai import OpenAI
import random
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from GridWorld import GridWorld
from utils import get_incremental_cost, count_transitions_until_target
client = OpenAI()

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
        capabilities= ("move","lift pallet"),
        gridworld=grid,
        initial_pos = start_pos,
        normal_ltl_spec=interest_points,
        T=T,
        actuation=actuation
    )
    agent_updated = GridWorldAgent(
        capabilities=("move","lift pallet"),
        gridworld=grid,
        initial_pos = start_pos,
        normal_ltl_spec=updated_interested_points,
        T=T,
        actuation=actuation
    )
    agent_original.solve()
    agent_updated.solve()
    path_original= agent_original.get_path()
    path_updated= agent_updated.get_path()
    t_h = count_transitions_until_target(path_updated,target_location=grid.conflict_cell)


    print(f"Agent {agent_original.id} path:", path_original)
    print(f"(Updated Agent {agent_updated.id} path:", path_updated)
    cmap = colors.ListedColormap(['white', 'gray', 'yellow', 'orange'])
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(grid_plot, cmap=cmap, origin='lower')

    if agent_original.model.Status == GRB.OPTIMAL:
        O_xs = [p[0] for p in path_original]
        O_ys = [p[1] for p in path_original]
        ax.plot(O_ys, O_xs, marker='o', color='red', label=f'Agent {agent_original.id} Path')
        ax.plot(start_pos[1], start_pos[0], marker='*', color='red', markersize=15)
        original_completion_time = int(agent_original.completion_time.X)

    if agent_updated.model.Status == GRB.OPTIMAL:
        U_xs = [p[0] for p in path_updated]
        U_ys = [p[1] for p in path_updated]
        ax.plot(U_ys, U_xs, marker='o', color='blue', label=f'Updated Agent {agent_updated.id} Path')
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
   





grid = GridWorld(grid_size=8,num_chunks=2,seed=31,num_dynamic_const=1)
print (f'LIST OF OBSTACLES{grid.obstacle_cells}')

compare_and_plot(15,grid,"A",2,1)

compare_and_plot(15,grid,'B',2,1)
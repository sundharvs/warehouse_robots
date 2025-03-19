from Robot import Robot
from GridWorld import GridWorld

world = GridWorld()

requester = Robot(
    normal_ltl_spec="A B C in 30",
    capabilities=["move"],
    initial_pos=(0, 0),
    gridworld=world,
    needs_help=True
)

num_helpers = 2
for n in range(num_helpers):
    robot = Robot(
        normal_ltl_spec="A B C in 30",
        capabilities=["lift pallet", "move"],
        initial_pos=(0, 0),
        gridworld=world,
        needs_help=False
    )
    
# Call this function at each timestep
world.update()
from Robot import GridWorldAgent
from GridWorld import GridWorld
from utils import start_pos_init, get_ltl_spec
world = GridWorld()



requester = GridWorldAgent(
    normal_ltl_spec=get_ltl_spec(grid=world,num_existing_locs=2,needs_help=True),
    capabilities=["move"],
    initial_pos=start_pos_init(world,True),
    gridworld=world,
    needs_help=True,
    T=15
)

num_helpers = 1
for n in range(num_helpers):
    robot = GridWorldAgent(
        normal_ltl_spec=get_ltl_spec(grid=world,num_existing_locs=2,needs_help=False),
        capabilities=["lift pallet", "move"],
        initial_pos=start_pos_init(world,False),
        gridworld=world,
        needs_help=False,
        T=15
    )
    
# Call this function at each timestep
world.update()

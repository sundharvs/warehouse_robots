from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import random
logger = logging.getLogger(__name__)


class GridWorld:
    '''GridWorld Class
    Parameters: 
    grid_size (n): number of grid on x and y-axis 
    num_chunks : number of aisle chunks, higher the number, more cut-throughs it will generate
    seed : this is used for random.seed, which is primarily used to change where the dynamic conflict will occur

    Attributes:
    free_cells : Type set => All (i,j) in the grid that is not occupied, including the dynmaic conflict grid
    obstacle_cells: Type set => All (i,j) in the grid that corresponds to obstacles known priori
    dynamic_conflict Type Dictionary, Default to one, {(i,j):"Scene description"} => Key: Tuple (i,j) in the grid where the requester robot starts & Value: String "VLM Scene Description" 
    '''
    foundation_model = "deepseek/deepseek-r1:free"
    
    def __init__(self, grid_size=12, num_chunks=4, seed=21, num_dynamic_const=1):
        self.robots = {}
        
        # World setup
        self.grid_size = grid_size
        self.num_chunks = num_chunks
        self.seed = seed
        self.num_dynamic_const = num_dynamic_const
        self.free_cells,self.obstacle_cells,self.conflict_cell,self.dynamic_conflict= self._create_grid_world()
        
        # Large Language Model setup
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        logging.basicConfig(filename='communication.log', level=logging.INFO)

    def _create_grid_world(self):
        random.seed(self.seed)

        grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        obstacle_cells = set()

        obstacle_cols = [2, self.grid_size - 3]

        num_whitespace_rows = self.num_chunks - 1
        total_obstacle_rows = self.grid_size - num_whitespace_rows
        chunk_height = total_obstacle_rows // self.num_chunks

        current_row = 0
        for chunk in range(self.num_chunks):
            for row in range(current_row, current_row + chunk_height):
                for col in obstacle_cols:
                    obstacle_cells.add((row, col))
            current_row += chunk_height + 1

        # Now explicitly handle negative indices:
        corrected_obstacle_cells = set()
        for (i, j) in obstacle_cells:
            corrected_i = i if i >= 0 else i + self.grid_size
            corrected_j = j if j >= 0 else j + self.grid_size
            # Only add corrected indices within valid range
            if 0 <= corrected_i < self.grid_size and 0 <= corrected_j < self.grid_size:
                corrected_obstacle_cells.add((corrected_i, corrected_j))

        # Fill obstacle cells into the grid safely
        for (i, j) in corrected_obstacle_cells:
            grid[i][j] = 1

        # Generate free cells explicitly excluding obstacle cells
        free_cells = [(i, j)
                    for i in range(self.grid_size)
                    for j in range(self.grid_size)
                    if (i, j) not in corrected_obstacle_cells]

        conflict_cell = random.sample(free_cells, self.num_dynamic_const)[0]

        vlm_scene_desc = ("Floor-level obstructions: The loose wooden pallet on the floor "
                        "and a cluster of boxes/pallets appear to partially block the aisle, "
                        "forcing any mobile robot or additional vehicles to navigate around them")

        dynamic_conflict = {conflict_cell: vlm_scene_desc}

        return free_cells, corrected_obstacle_cells, conflict_cell, dynamic_conflict

    def register_robot(self, robot):
        """Register a new robot in the gridworld."""
        robot_id = len(self.robots)
        self.robots[robot_id] = robot
        return robot_id

    def broadcast_message(self, sender_id, message):
        """Send a message from one robot to all others."""
        print('message broadcasted')
        logging.info(f"{sender_id} => all: {message}")
        for robot_id, robot in self.robots.items():
            if robot_id != sender_id:
                robot.receive_message(sender_id, message)
                
    def send_message(self, sender_id, receiver_id, message):
        """Send a message from one robot to another."""
        print('message sent')
        logging.info(f"{sender_id} => {receiver_id}: {message}")
        self.robots[receiver_id].receive_message(sender_id, message)
    
    def update(self, positions=None):
        """Update the gridworld state at each timestep."""
        for robot_id, robot in self.robots.items():
            # robot.position = positions[robot_id]
            # TODO: more Robot object atrribute updates
            if robot.needs_help:
                robot.generate_help_request("A pallet is blocking the aisle.")
    
    def llm_query(self, prompt):
        """Query the LLM with a given prompt."""
        completion = self.llm_client.chat.completions.create(model=self.foundation_model, messages=[{"role": "user", "content": prompt}])
        return completion.choices[0].message.content
        # TODO: structured outputs / json mode
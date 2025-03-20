import random


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

    def __init__(self,grid_size=12,num_chunks=4,seed=21,num_dynamic_const=1):
        self
        self.grid_size = grid_size
        self.num_chunks = num_chunks
        self.seed = seed
        self.num_dynamic_const = num_dynamic_const
        self.free_cells,self.obstacle_cells,self.conflict_cell,self.dynamic_conflict= self._create_grid_world()
        



    # def _create_grid_world(self):
    #     '''Initalizes the grid world with three types of occupancy
    #         free_cells = 0 , obstaacle_cells (static obstacles known priori) = 1, dynamic confict (obstacles encountered by VLM at runtime)=2

    #         Returns:
    #         free_cells : Type set => All (i,j) in the grid that is not occupied, including the dynmaic conflict grid
    #         obstacle_cells: Type set => All (i,j) in the grid that corresponds to obstacles known priori
    #         conflict_cell: Type tuple => (i,j) in the grid that is the location of the dynamic_conflict
    #         dynamic_conflict Type Dictionary, Default to one, {(i,j):"Scene description"} => Key: Tuple (i,j) in the grid where the requester robot starts & Value: String "VLM Scene Description" 
    #     '''

    #     # Initialize grid with all free spaces (0)
    #     #create all free spaces of nxn 
    #     random.seed(self.seed)  # for reproducibility
    

    #     grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
    #     #Create obstacle_cells
    #     # obstacle_cells = set()
    #     # for n in range(1,self.num_chunks+1):
    #     #     for j in range(2,self.grid_size,2):
    #     #         for i in range (int((n-1)*self.grid_size/self.num_chunks)+1,int(n*self.grid_size/self.num_chunks)):
    #     #             obstacle_cells.add((i,j))

    #     # obstacle_cells = set()
    #     # for n in range(1, self.num_chunks + 1):  # start from 1
    #     #     for j in range(2, self.grid_size, 2):
    #     #         start_i = int((n - 1) * self.grid_size / self.num_chunks)
    #     #         end_i = int(n * self.grid_size / self.num_chunks)
    #     #         for i in range(start_i, end_i):
    #     #             obstacle_cells.add((i, j))
    #     obstacle_cells = set()
    #     obstacle_cols = range(2, self.grid_size - 2)

    #     # Compute chunk height and remaining rows
    #     num_whitespace_rows = self.num_chunks - 1
    #     total_obstacle_rows = self.grid_size - num_whitespace_rows
    #     chunk_height = total_obstacle_rows // self.num_chunks

    #     current_row = 0
    #     for chunk in range(self.num_chunks):
    #         # For each chunk, add obstacle cells vertically
    #         for row in range(current_row, current_row + chunk_height):
    #             for col in obstacle_cols:
    #                 obstacle_cells.add((row, col))
    #         current_row += chunk_height + 1  # Skip one row for whitespace


            
    #         for (i,j) in obstacle_cells:
    #             grid[i][j]=1
        
    #     free_cells = [(i, j) 
    #           for i in range(self.grid_size) 
    #           for j in range(self.grid_size) 
    #           if (i, j) not in obstacle_cells]
    #     conflict_cell = random.sample(free_cells, self.num_dynamic_const)[0]
    #     vlm_scene_desc = "Floor-level obstructions: The loose wooden pallet on the floor and a cluster of boxes/pallets appear to partially \
    #                         block the aisle, forcing any mobile robot or additional vehicles to navigate around them"
    #     dynamic_conflict = {conflict_cell:vlm_scene_desc}
    #     return free_cells, obstacle_cells,conflict_cell,dynamic_conflict
    
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


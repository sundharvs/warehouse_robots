from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class GridWorld:
    model = "deepseek/deepseek-r1:free"
    
    def __init__(self):
        self.robots = {}
        
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        logging.basicConfig(filename='communication.log', level=logging.INFO)

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
        completion = self.llm_client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
        return completion.choices[0].message.content
        # TODO: structured outputs / json mode
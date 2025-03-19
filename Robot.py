from GridWorld import GridWorld
from Message import Message
from openai import OpenAI
import logging

class Robot:
    def __init__(self, normal_ltl_spec, capabilities, initial_pos, gridworld, needs_help=False):
        self.ltl_spec = normal_ltl_spec
        self.capabilities = capabilities
        self.position = initial_pos
        self.gridworld = gridworld
        self.id = self.gridworld.register_robot(self)
        self.needs_help = needs_help
        
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
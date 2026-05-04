import typing

import pacai.agents.greedy
import pacai.agents.minimax
import pacai.core.action
import pacai.core.gamestate
from pacai.search.distance import manhattan_distance

class ReflexAgent(pacai.agents.greedy.GreedyAgent):
    """
    A simple agent based on pacai.agents.greedy.GreedyAgent.

    You job is to make this agent better (it is pretty bad right now).
    You can change whatever you want about it,
    but it should still be a child of pacai.agents.greedy.GreedyAgent
    and be a "reflex" agent.
    This means that it shouldn't do any formal planning or searching,
    instead it should just look at the state of the game and try to make a good choice in the moment.
    You can make a great agent just by implementing a custom evaluate_state() method
    (and maybe add to the constructor if you want).
    """

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        # Put code here if you want.

    def evaluate_state(self,
            state: pacai.core.gamestate.GameState,
            action: pacai.core.action.Action | None = None,
            **kwargs: typing.Any) -> float:
        # This would be a great place to improve your reflex agent.
        
        #need location of pacman
        pacman_pos = state.get_agent_position(0) #pac is agent 0
        if pacman_pos is None: #he dead rip
            return -float('inf') # BAD
        score = state.score
        
        food = state.get_food() #set of food locations
        if food: #set not empty
            closest_food = float('inf')
            for f in food: #individual pos
                mhd = manhattan_distance(pacman_pos, f)
                if mhd < closest_food:
                    closest_food = mhd
            score += 1.0 / closest_food #closer food (small num) -> higher score, better, we wat to go there!
    
        for pos in state.get_nonscared_ghost_positions().values(): #returns dict
            dist = manhattan_distance(pacman_pos, pos)
            if dist <= 1:
                score -= 500 #if it's super close, we gonna die, DONT GO HERE
            else:
                score -= 10.0 / dist  #10 cause we want to care more about avoiding ghosts than eating food. if it's between the two, choose life duh
        
        for pos in state.get_scared_ghost_positions().values(): #returns dict
            dist = manhattan_distance(pacman_pos, pos)
            #if close to scared ghost, add points it's good. add 1 to denom in case we're at dist = 0 (current state is eating ghost)
            score += 50 / (dist + 1)

        #finally, food left, cuz we wanna get to end goal state
        score -= 4 * len(food)
            
        return score

class MyMinimaxLikeAgent(pacai.agents.minimax.MinimaxLikeAgent):
    """
    An agent that implements all the required methods for the minimax family of algorithms.
    Default implementations are supplied, so the agent should run right away,
    but it will not be very good.

    To implement minimax, minimax_step_max() and minimax_step_min() are required
    (you can ignore alpha and beta).

    To implement minimax with alpha-beta pruning,
    minimax_step_max() and minimax_step_min() with alpha and beta are required.

    To implement expectimax, minimax_step_max() and minimax_step_expected_min() are required.

    You are free to implement/override any methods you need to.
    """

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        # You can use the constructor if you need to.

#base class hamndles whether to go to max or min based on agent. i need to implement the minimax algo for each
    def minimax_step_max(self,
            state: pacai.core.gamestate.GameState,
            ply_count: int,
            legal_actions: list[pacai.core.action.Action],
            alpha: float,
            beta: float,
            ) -> tuple[list[pacai.core.action.Action], float]:
        # *** Your Code Here ***
        #return list of actions and score
        
        #if terminal node, return utility value
        #
        best_actions = []
        v = -float('inf')
        
        #now how to add alphabeta?
        
        
        #basically, if the next thing is larger, make it best score (v) and make best_actions = a list of that action, cuz minimax get_action() will take randomly from actions list
        #if they equal, we append them because it again, chooses randomly (this is what breaks the tie)
        
        for action in legal_actions:
            successor = state.generate_successor(action, self.rng)
            _, score = self.minimax_step(successor, ply_count, alpha, beta)
            #call recursively to get score
            if score > v:
                v = score
                best_actions = [action] # a new list, because we've found a new max so overwrite wtv we thought better
            if score == v:
                best_actions.append(action) 
        
        return best_actions, v
    
    

    def minimax_step_min(self,
            state: pacai.core.gamestate.GameState,
            ply_count: int,
            legal_actions: list[pacai.core.action.Action],
            alpha: float,
            beta: float,
            ) -> tuple[list[pacai.core.action.Action], float]:
        best_actions = []
        v = float('inf')
        
        #basically, if the next thing is larger, make it best score (v) and make best_actions = a list of that action, cuz minimax get_action() will take randomly from actions list
        #if they equal, we append them because it again, chooses randomly (this is what breaks the tie)
        
        for action in legal_actions:
            successor = state.generate_successor(action, self.rng)
            _, score = self.minimax_step(successor, ply_count, alpha, beta)
            #call recursively to get score
            if score < v:
                v = score
                best_actions = [action] # a new list, because we've found a new max so overwrite wtv we thought better
            if score == v:
                best_actions.append(action) 
        
        return best_actions, v
    
    
        return super().minimax_step_min(state, ply_count, legal_actions, alpha, beta)

    def minimax_step_expected_min(self,
            state: pacai.core.gamestate.GameState,
            ply_count: int,
            legal_actions: list[pacai.core.action.Action],
            alpha: float,
            beta: float,
            ) -> float:
        # *** Your Code Here ***
        return super().minimax_step_expected_min(state, ply_count, legal_actions, alpha, beta)

def better_state_eval(
        state: pacai.core.gamestate.GameState,
        agent: typing.Any | None = None,
        action: pacai.core.action.Action | None = None,
        **kwargs: typing.Any) -> float:
    """
    Create a better state evaluation function for your MyMinimaxLikeAgent agent!

    In this comment, include a description of what you are doing.

    *** Your Text Here ***
    """

    # *** Your Code Here ***
    return state.score

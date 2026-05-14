"""
In this file, you will implement code relating to simple single-agent searches.
"""

import random
import typing

import pacai.agents.searchproblem
import pacai.core.agent
import pacai.core.agentaction
import pacai.core.board
import pacai.core.gamestate
import pacai.core.search
import pacai.pacman.board
import pacai.search.common
import pacai.search.food
import pacai.search.position
import pacai.util.containers

def depth_first_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements depth first search (DFS).
    This means that it will search the deepest nodes in the search tree first.
    See: https://en.wikipedia.org/wiki/Depth-first_search .
    """

    start = problem.get_starting_node()  # start for DFS
    visited = set()
    # add to stack
    # stack needs to store the actions/path and cost at each point
    stack = [(start, [], 0.0)]  # node, path, cost

    while stack:  # not empty

        node, actions, cost = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        if problem.is_goal_node(node):
            problem.complete(node)
            return pacai.core.search.SearchSolution(actions, cost, node)  # the solution we found

        for successor in problem.get_successor_nodes(node):
            if successor.node not in visited:
                stack.append((successor.node, actions + [successor.action], cost + successor.cost))

    # DFS implementatino:

    # for x in start.nieghbors
    #     if not visited[x]
    #         visited[x]
    #         add successor nodes to stack

    # for each one, we add the node to the stack, track action and cost

    # return a soltuion of actions, goal node, cost
    # is cost the same on all edges? how to know what it is?
    # how to track actions?

    raise pacai.core.search.SolutionNotFoundError()

def breadth_first_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements breadth first search (BFS).
    This means that it will search nodes based on what level in search tree they appear.
    See: https://en.wikipedia.org/wiki/Breadth-first_search .
    """

    start = problem.get_starting_node()  # start for BFS
    visited = set()
    queue = pacai.util.containers.Queue()
    queue.push((start, [], 0.0))

    while not queue.is_empty():  # not empty

        node, actions, cost = queue.pop()
        if node in visited:
            continue
        visited.add(node)

        if problem.is_goal_node(node):
            problem.complete(node)
            return pacai.core.search.SearchSolution(actions, cost, node)  # the solution we found

        for successor in problem.get_successor_nodes(node):
            if successor.node not in visited:
                queue.push((successor.node, actions + [successor.action], cost + successor.cost))

    raise pacai.core.search.SolutionNotFoundError()

def uniform_cost_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements uniform cost search (UCS).
    This means that it will search nodes with a lower total cost first.
    See: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Practical_optimizations_and_infinite_graphs .
    """
    start = problem.get_starting_node()  # start
    visited = set()
    queue = pacai.util.containers.PriorityQueue()
    queue.push((start, [], 0.0), 0.0)  # added it's # to compare with

    while not queue.is_empty():  # not empty

        node, actions, cost = queue.pop()
        if node in visited:
            continue
        visited.add(node)

        if problem.is_goal_node(node):
            problem.complete(node)
            return pacai.core.search.SearchSolution(actions, cost, node)  # the solution we found

        for successor in problem.get_successor_nodes(node):
            if successor.node not in visited:
                new_cost = cost + successor.cost
                queue.push((successor.node, actions + [successor.action], new_cost), new_cost)

    raise pacai.core.search.SolutionNotFoundError()


def astar_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements A* search (pronounced "A Star search").
    This means that it will search nodes with a lower combined cost and heuristic first.
    See: https://en.wikipedia.org/wiki/A*_search_algorithm .
    """

    start = problem.get_starting_node()  # start
    visited = set()
    queue = pacai.util.containers.PriorityQueue()
    queue.push((start, [], 0.0), 0.0)  # now 0.0 will be g(n) + h(n). add a heuristic somewhere

    while not queue.is_empty():  # not empty

        node, actions, cost = queue.pop()
        if node in visited:
            continue
        visited.add(node)

        if problem.is_goal_node(node):
            problem.complete(node)
            return pacai.core.search.SearchSolution(actions, cost, node)  # the solution we found

        for successor in problem.get_successor_nodes(node):
            if successor.node not in visited:
                new_cost = cost + successor.cost  # cost to reach node
                priority = new_cost + heuristic(successor.node, problem)  # estimated cost to goal, prioritize by that
                queue.push((successor.node, actions + [successor.action], new_cost), priority)

    raise pacai.core.search.SolutionNotFoundError()


class CornersSearchNode(pacai.core.search.SearchNode):
    """
    A search node the can be used to represent the corners search problem.

    You get to implement this search node however you want.
    """

    def __init__(self, position: pacai.core.board.Position, corners_visited: frozenset) -> None:
        """ Construct a search node to help search for corners. """
        self.position: pacai.core.board.Position = position
        self.corners_visited = corners_visited  # set of corners visited. frozen so it's hashable bc we need to add it to visited set later

    def __lt__(self, other: object) -> bool:
        if (not isinstance(other, CornersSearchNode)):
            return False

        return (self.position < other.position)

    def __eq__(self, other: object) -> bool:
        if (not isinstance(other, CornersSearchNode)):
            return False

        return (self.position == other.position and self.corners_visited == other.corners_visited)

    # checking equivalency - given another node, we'll see if it's same posiiton/same corner
    def __hash__(self) -> int:
        return hash((self.position, self.corners_visited))


class CornersSearchProblem(pacai.core.search.SearchProblem[CornersSearchNode]):
    """
    A search problem for touching the four different corners in a board.

    You may assume that very board is surrounded by walls (e.g., (0, 0) is a wall),
    and that the position diagonally inside from the walled corner is the location we are looking for.
    For example, if we had a square board that was 10x10, then we would be looking for the following corners:
     - (1, 1) -- North-West / Upper Left
     - (1, 8) -- North-East / Upper Right
     - (8, 1) -- South-West / Lower Left
     - (8, 8) -- South-East / Lower Right
    """

    def __init__(self,
            game_state: pacai.core.gamestate.GameState,
            cost_function: pacai.core.search.CostFunction | str = pacai.util.alias.COST_FUNC_UNIT.long,
            **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        self.board = game_state.board
        self.start_position = game_state.get_agent_position()
        self.corners = frozenset(self.board.get_corners(offset=1))  # corners!

        if (isinstance(cost_function, str)):
            cost_function = typing.cast(pacai.core.search.CostFunction, pacai.util.reflection.fetch(cost_function))
        self._cost_function: pacai.core.search.CostFunction = cost_function

    def get_starting_node(self) -> CornersSearchNode:
        return CornersSearchNode(self.start_position, frozenset())

    def is_goal_node(self, node: CornersSearchNode) -> bool:
        # *** Your Code Here *** #this will be: if the 4 corners were reached?
        return node.corners_visited == self.corners

    def get_successor_nodes(self, node: CornersSearchNode) -> list[pacai.core.search.SuccessorInfo]:
        successors = []

        # Check all the non-wall neighbors.
        for (action, position) in self.board.get_neighbors(node.position):
            new_corners = node.corners_visited
            # for all neighbors, we'll see wut the corners are and then update them
            if position in self.corners:
                new_corners = node.corners_visited | frozenset({position})
            # if the positino is a corner, mark it in corners_visited

            next_node = CornersSearchNode(position, new_corners)  # new node. state only holds pos, and bool of coerns viitsted
            cost = self._cost_function(next_node)  # find cost of next node
            # append this
            successors.append(pacai.core.search.SuccessorInfo(next_node, action, cost))

        # Do bookkeeping on the states/positions we visited.
        self.expanded_node_count += 1
        if (node not in self.visited_nodes):
            self.position_history.append(node.position)

        return successors


def corners_heuristic(node: CornersSearchNode, problem: CornersSearchProblem, **kwargs: typing.Any) -> float:
    """
    A heuristic for CornersSearchProblem.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """
    # i think manhattan would be good, because it can never overestimate (cuz pacman moves in a grid). so manhattan is SP between two points,
    # and actual cost will either be >=

    unvisited = list(problem.corners - node.corners_visited)  # can u do that w list idk yet
    # total coerners - ones alr visited at this node (state)

    if not unvisited:  # no corners left
        return 0.0

    total = 0.0
    curr = node.position

    # manhattan distance is abs(row-row) + abs(col-col)

    while unvisited:  # iterate thru
        nearest_corner = None
        nearest_distance = 99999999
        for corner in unvisited:
            mhd = abs(curr.row - corner.row) + abs(curr.col - corner.col)  # distance from current node position to each corner
            if mhd < nearest_distance:
                nearest_distance = mhd
                nearest_corner = corner

        total += nearest_distance
        curr = nearest_corner  # now this is our position, we're looking for the nearest corner from that one
        unvisited.remove(nearest_corner)

    return total  # numeric distance, but it's all a sum is that good idk

def food_heuristic(node: pacai.search.food.FoodSearchNode, problem: pacai.search.food.FoodSearchProblem, **kwargs: typing.Any) -> float:
    """
    A heuristic for the FoodSearchProblem.
    """
    # start by fa
    unvisited = list(node.remaining_food)

    if not unvisited:
        return 0.0

    far = 0.0

    for food in unvisited:
        mhd = abs(node.position.row - food.row) + abs(node.position.col - food.col)  # distance from current node position to each corner
        if mhd > far:
            far = mhd

    return far

    """
    curr = node.position

    while unvisited:
        nearest_food = None
        nearest_distance = 999999
        for food in unvisited:
            mhd = abs(curr.row-food.row) + abs(curr.col-food.col) #distance from current node position to each corner
            if mhd < nearest_distance:
                nearest_distance = mhd
                nearest_food = food

        total += nearest_distance
        curr = nearest_food #now this is our position, we're looking for the nearest corner from that one
        unvisited.remove(nearest_food)

    return total
"""

class ClosestDotSearchAgent(pacai.agents.searchproblem.GreedySubproblemSearchAgent):
    """
    Search for a path to all the food by greedily searching for the next closest food again and again
    (util we have reached all the food).

    This agent is left to you to fill out.
    But make sure to take your time and think.
    The final solution is quite simple if you take your time to understand everything up until this point and leverage
    pacai.agents.searchproblem.GreedySubproblemSearchAgent and pacai.student.problem.AnyMarkerSearchProblem.
    pacai.agents.searchproblem.GreedySubproblemSearchAgent is already implemented,
    but you should take some time to understand it.
    pacai.student.problem.AnyMarkerSearchProblem (below in this file) has not yet been implemented,
    but is the quickest and easiest way to implement this class.

    Hint:
    Remember that you can call a parent class' `__init__()` method from a child class' `__init__()` method.
    (See pacai.student.problem.AnyMarkerSearchProblem for an example.)
    Child classes will generally always call their parent's `__init__()` method
    (if a child class does not implement `__init__()`, then the parent's `__init__()` is automatically called).
    This call does not need to be the first line in the method,
    and you can pass whatever you want to the parent's `__init__()`.
    """
    # now just tell it to use..... anymarker
    def __init__(self, **kwargs):
        super().__init__(problem=AnyMarkerSearchProblem, solver=pacai.util.alias.SEARCH_SOLVER_BFS.long, **kwargs)
    # *** Your Code Here ***

class AnyMarkerSearchProblem(pacai.search.position.PositionSearchProblem):
    """
    A search problem for finding a path to any instance of the specified board marker (e.g., food, wall, power capsule).

    This search problem is just like the pacai.search.position.PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    You may modify the `__init__()` if you want, the other methods should be fine as-is.
    """

    def __init__(self,
            game_state: pacai.core.gamestate.GameState,
            target_marker: pacai.core.board.Marker = pacai.pacman.board.MARKER_PELLET,
            **kwargs: typing.Any) -> None:
        super().__init__(game_state, **kwargs)

        self.target_marker = target_marker

        # want it to determine whether we foun food, and this will bevocme the prbolem that greedy solves

    def is_goal_node(self, node: pacai.search.position.PositionSearchNode) -> bool:
        # *** Your Code Here ***
        return self.board.is_marker(self.target_marker, node.position)

class ApproximateSearchAgent(pacai.agents.searchproblem.GreedySubproblemSearchAgent):
    """
    A search agent that tries to perform an approximate search instead of an exact one.
    In other words, this agent is okay with a solution that is "good enough" and not necessarily optimal.
    """
    def __init__(self, **kwargs):
        super().__init__(problem=AnyMarkerSearchProblem, solver=pacai.util.alias.SEARCH_SOLVER_BFS.long, **kwargs)

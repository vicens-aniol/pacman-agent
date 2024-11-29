# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
import math
import heapq
import time


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAStarAgent', second='DefensiveAStarAgent', num_training=0):
    """
    Creates a team of two agents: one offensive and one defensive.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class OffensiveAStarAgent(CaptureAgent):
    """
    Offensive Agent that uses A* for pathfinding towards the nearest food,
    avoids enemies, and returns to base when carrying sufficient food or being pursued.
    """

    def register_initial_state(self, game_state):
        """
        initializes the agent and precomputes maze distances
        """
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.distancer.get_maze_distances()
        self.max_carry = 6  # define when to return to base

    def is_on_own_side(self, game_state, position):
        """
        checks if the agent is on its own side of the board
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        if self.red:
            return position[0] < mid_x
        else:
            return position[0] >= mid_x

    def choose_action(self, game_state):
        """
        picks the best action based on A* pathfinding towards the nearest food or returning to base
        if a ghost is very close, returns to base immediately avoiding all ghosts
        """
        actions = game_state.get_legal_actions(self.index)
        actions = [action for action in actions if action != Directions.STOP]

        if not actions:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        my_state = game_state.get_agent_state(self.index)
        carrying_food = my_state.num_carrying

        # check if the agent is being pursued
        if self.is_being_pursued(game_state):
            target = self.get_closest_home_position(game_state, my_pos)
            mode = 'return'
        else:
            # agent's state: collect food or return to base
            if carrying_food >= self.max_carry:
                # mode: return to base
                target = self.get_closest_home_position(game_state, my_pos)
                mode = 'return'
            else:
                # mode: collect food
                if food_list:
                    target = self.get_closest_food(game_state, my_pos, food_list)
                    mode = 'collect'
                else:
                    # no food left
                    return Directions.STOP

        next_step = self.a_star_search_next_step(game_state, my_pos, target, mode)

        if next_step:
            action = self.get_action_from_path(my_pos, next_step)
            return action
        else:
            return Directions.STOP

    def is_being_pursued(self, game_state):
        """
        checks if there are enemy ghosts very close chasing the agent
        returns True if there's at least one ghost within the safety distance
        """
        my_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_enemies = [enemy for enemy in enemies if not enemy.is_pacman and enemy.get_position() is not None]

        for enemy in active_enemies:
            enemy_pos = enemy.get_position()
            distance = self.get_maze_distance(my_pos, enemy_pos)
            if distance <= 2:
                return True
        return False

    def get_closest_food(self, game_state, my_pos, food_list):
        """
        finds the closest food that's not too close to enemies
        """
        safe_food = []
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_enemies = [enemy for enemy in enemies if not enemy.is_pacman and enemy.get_position() is not None]

        for food in food_list:
            is_safe = True
            for enemy in active_enemies:
                enemy_pos = enemy.get_position()
                distance = self.get_maze_distance(my_pos, enemy_pos)
                if distance <= 4:  # define a safety distance
                    is_safe = False
                    break
            if is_safe:
                safe_food.append(food)

        if safe_food:
            closest_food = min(safe_food, key=lambda food: self.get_maze_distance(my_pos, food))
        else:
            # if no safe food, pick the closest one anyway
            closest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
        return closest_food

    def get_closest_home_position(self, game_state, my_pos):
        """
        finds the closest position on the home boundary
        """
        home_positions = self.get_home_boundary_positions(game_state)
        if not home_positions:
            return self.start  # fallback to the starting position
        closest_home = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
        return closest_home

    def a_star_search_next_step(self, game_state, start, goal, mode):
        """
        A* search algorithm that returns only the next step towards the goal
        """
        walls = game_state.get_walls()
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        open_set = util.PriorityQueue()
        open_set.push(start, 0)
        came_from = {}
        cost_so_far = {start: 0}

        nodes_expanded = 0
        MAX_NODES = 60000
        start_time_search = time.time()
        TIME_LIMIT_SEARCH = 0.5  # seconds

        while not open_set.is_empty():
            current = open_set.pop()
            nodes_expanded += 1

            # check node and time limits
            if nodes_expanded > MAX_NODES:
                return None

            if time.time() - start_time_search > TIME_LIMIT_SEARCH:
                return None

            if current == goal:
                break

            for next_pos in self.get_successors(current, walls):
                new_cost = cost_so_far[current] + self.get_cost(game_state, next_pos, mode)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal, mode)
                    open_set.push(next_pos, priority)
                    came_from[next_pos] = current

        # reconstruct the path
        if goal in came_from or current == goal:
            path = [goal]
            current_step = goal
            steps = 0
            MAX_PATH_STEPS = 1000

            while current_step != start and steps < MAX_PATH_STEPS:
                current_step = came_from.get(current_step, start)
                path.append(current_step)
                steps += 1

            if current_step == start:
                path.reverse()
                if len(path) > 1:
                    next_step = path[1]  # first step after the start
                    return next_step
                else:
                    return None
            else:
                return None
        else:
            return None  # No path found

    def get_cost(self, game_state, position, mode):
        """
        calculates the cost of moving to a given position
        increases the cost if the position is near enemies, only if the agent is on the enemy's side
        """
        cost = 1  # base cost

        # determine if the agent is on its own side
        if self.is_on_own_side(game_state, position):
            # no penalize for enemies on own side
            pass
        else:
            # on enemy's side, penalize positions near enemies
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            active_enemies = [enemy for enemy in enemies if not enemy.is_pacman and enemy.get_position() is not None]

            for enemy in active_enemies:
                enemy_pos = enemy.get_position()
                distance = self.get_maze_distance(position, enemy_pos)
                if distance <= 1:
                    return math.inf  # avoid positions adjacent to enemies
                elif distance == 2:
                    cost += 50   # high penalty to maintain a safe distance
                elif 3 <= distance <= 4:
                    cost += 10   # moderate penalty for greater distances

        # if in 'return' mode, add cost based on distance from home
        if mode == 'return':
            home_positions = self.get_home_boundary_positions(game_state)
            if home_positions:
                min_home_distance = min([self.get_maze_distance(position, pos) for pos in home_positions])
                cost += min_home_distance * 0.2  # add cost for being farther from home

        return cost  # ensure the method returns the calculated cost

    def get_successors(self, position, walls):
        """
        returns a list of successor positions from the current position
        """
        successors = set()
        x, y = position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_x = x + dx
            next_y = y + dy
            if 0 <= next_x < walls.width and 0 <= next_y < walls.height and not walls[next_x][next_y]:
                successors.add((next_x, next_y))
        return list(successors)

    def heuristic(self, a, b, mode):
        """
        Manhattan distance heuristic for A*
        """
        (x1, y1) = a
        (x2, y2) = b
        distance = abs(x1 - x2) + abs(y1 - y2)

        if mode == 'return':
            # adjust the heuristic to prioritize returning home
            return distance * 1.2  # increase the heuristic value to prefer shorter paths home
        else:
            return distance

    def get_action_from_path(self, current_pos, next_pos):
        """
        determines the action needed to move from current_pos to next_pos
        """
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        if dx == 1:
            return Directions.EAST
        elif dx == -1:
            return Directions.WEST
        elif dy == 1:
            return Directions.NORTH
        elif dy == -1:
            return Directions.SOUTH
        else:
            return Directions.STOP

    def get_home_boundary_positions(self, game_state):
        """
        returns a list of positions on the home boundary
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2 # '//' because we want integer division
        if self.red:
            home_x = mid_x - 1
        else:
            home_x = mid_x
        home_positions = [(home_x, y) for y in range(walls.height) if not walls[home_x][y]]
        return home_positions

    def get_maze_distance(self, pos1, pos2):
        """
        computes the maze distance between two points using the Distancer
        """
        try:
            distance = self.distancer.get_distance(pos1, pos2)
            if distance is None:
                return math.inf
            return distance
        except Exception as e:
            return math.inf

class DefensiveAStarAgent(CaptureAgent):
    """
    Defensive Agent that uses A* for pathfinding to intercept invaders and protect own maze.
    """

    def register_initial_state(self, game_state):
        """
        initializes the agent and precomputes maze distances
        """
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.distancer.get_maze_distances()
        self.max_carry = 6  # define when to return to base if necessary

    def choose_action(self, game_state):
        """
        picks the best defensive action based on A* pathfinding to intercept invaders
        """
        actions = game_state.get_legal_actions(self.index)
        actions = [action for action in actions if action != Directions.STOP]

        if not actions:
            return Directions.STOP

        my_pos = game_state.get_agent_position(self.index)
        invaders = self.get_invaders(game_state)

        if invaders:
            # if there are invaders, intercept them
            closest_invader = self.get_closest_invader(my_pos, invaders)
            target = closest_invader
            mode = 'intercept'
        else:
            # if no invaders, patrol near the base
            target = self.get_patrol_position(game_state)
            mode = 'patrol'

        next_step = self.a_star_search_next_step(game_state, my_pos, target, mode)

        if next_step:
            action = self.get_action_from_path(my_pos, next_step)
            return action
        else:
            return Directions.STOP

    def get_invaders(self, game_state):
        """
        gets a list of enemies acting as Pacmen (invaders) on our side
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [enemy for enemy in enemies if enemy.is_pacman and enemy.get_position() is not None]
        return invaders

    def get_closest_invader(self, my_pos, invaders):
        """
        finds the closest invader to the agent
        """
        distances = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
        min_distance = min(distances)
        closest_invader = invaders[distances.index(min_distance)]
        return closest_invader.get_position()

    def get_patrol_position(self, game_state):
        """
        defines a patrol position near the base
        """
        home_positions = self.get_home_boundary_positions(game_state)
        if not home_positions:
            return self.start
        # randomly pick a patrol position
        return random.choice(home_positions)

    def a_star_search_next_step(self, game_state, start, goal, mode):
        """
        performs A* search and returns the next step towards the goal
        """
        walls = game_state.get_walls()
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        open_set = util.PriorityQueue()
        open_set.push(start, 0)
        came_from = {}
        cost_so_far = {start: 0}

        nodes_expanded = 0
        MAX_NODES = 60000 # to avoid memory overflow
        start_time_search = time.time()
        TIME_LIMIT_SEARCH = 0.5  # seconds

        while not open_set.is_empty():
            current = open_set.pop()
            nodes_expanded += 1

            # check node and time limits
            if nodes_expanded > MAX_NODES:
                return None

            if time.time() - start_time_search > TIME_LIMIT_SEARCH:
                return None

            if current == goal:
                break

            for next_pos in self.get_successors(current, walls):
                new_cost = cost_so_far[current] + self.get_cost(game_state, next_pos, mode)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal, mode)
                    open_set.push(next_pos, priority)
                    came_from[next_pos] = current

        # reconstruct the path
        if goal in came_from or current == goal:
            path = [goal]
            current_step = goal
            steps = 0
            MAX_PATH_STEPS = 1000  # limit steps to avoid infinite loops

            while current_step != start and steps < MAX_PATH_STEPS:
                current_step = came_from.get(current_step, start)
                path.append(current_step)
                steps += 1

            if current_step == start:
                path.reverse()
                if len(path) > 1:
                    next_step = path[1]  # first step after the start
                    return next_step
                else:
                    return None
            else:
                return None
        else:
            return None  # No path found

    def get_cost(self, game_state, position, mode):
        """
        calculates the cost of moving to a given position
        lower cost when approaching an invader, higher cost if near non-invader enemies
        """
        cost = 1  # base cost

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_enemies = [enemy for enemy in enemies if not enemy.is_pacman and enemy.get_position() is not None]

        if mode == 'intercept':
            # lower cost to approach invaders
            pass  # do not increase additional cost
        elif mode == 'patrol':
            # higher cost near enemies to avoid unnecessary confrontations
            for enemy in active_enemies:
                enemy_pos = enemy.get_position()
                distance = self.get_maze_distance(position, enemy_pos)
                if distance <= 1:
                    return math.inf  # avoid positions adjacent to enemies
                elif distance == 2:
                    cost += 50  # high penalty to maintain a safe distance
                elif 3 <= distance <= 4:
                    cost += 10  # moderate penalty

        return cost

    def get_successors(self, position, walls):
        """
        returns a list of valid successor positions from the current position
        """
        successors = set()
        x, y = position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_x = x + dx
            next_y = y + dy
            if 0 <= next_x < walls.width and 0 <= next_y < walls.height and not walls[next_x][next_y]:
                successors.add((next_x, next_y))
        return list(successors)

    def heuristic(self, a, b, mode):
        """
        Manhattan distance heuristic for A*
        """
        (x1, y1) = a
        (x2, y2) = b
        distance = abs(x1 - x2) + abs(y1 - y2)

        if mode == 'intercept':
            # prioritize shorter paths towards invaders
            return distance
        elif mode == 'patrol':
            # adjust the heuristic to prefer paths that avoid enemies
            return distance * 1.2  # increase the value to prefer shorter paths to patrol
        else:
            return distance

    def get_action_from_path(self, current_pos, next_pos):
        """
        determines the action needed to move from current_pos to next_pos
        """
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        if dx == 1:
            return Directions.EAST
        elif dx == -1:
            return Directions.WEST
        elif dy == 1:
            return Directions.NORTH
        elif dy == -1:
            return Directions.SOUTH
        else:
            return Directions.STOP

    def get_home_boundary_positions(self, game_state):
        """
        returns a list of positions on the home boundary
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        if self.red:
            home_x = mid_x - 1
        else:
            home_x = mid_x
        home_positions = [(home_x, y) for y in range(walls.height) if not walls[home_x][y]]
        return home_positions

    def get_maze_distance(self, pos1, pos2):
        """
        calculates the maze distance between two points using the Distancer
        """
        distance = self.distancer.get_distance(pos1, pos2)
        if distance is None:
            return math.inf
        return distance
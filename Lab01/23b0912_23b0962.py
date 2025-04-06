import numpy as np
from collections import deque
import heapq
from typing import List, Tuple, Set, Dict  # noqa: F401
"""
Do not import any other package unless allowed by te TAs in charge of the lab.
Do not change the name of any of the functions below.
"""

# List of possible moves (up, down, left, right)
directions = [(-1, 0, 'U'), (1, 0, 'D'), (0, -1, 'L'), (0, 1, 'R')]

# State Transition (Move Generation)

def get_neighbors(state):
    neighbors = []
    blank_pos = state.index(0)
    row, col = blank_pos // 3, blank_pos % 3  # row and column of the blank tile

    for dr, dc, move in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_blank_pos = new_row * 3 + new_col
            # Swap the blank tile with the new position
            new_state = list(state)
            new_state[blank_pos], new_state[new_blank_pos] = new_state[new_blank_pos], new_state[blank_pos]
            neighbors.append((tuple(new_state), move))
    
    return neighbors

# Manhattan Distance Heuristic
def manhattan_heuristic(state, goal):
    distance = 0
    for i in range(9):
        if state[i] != 0:  # ignore the blank tile
            goal_pos = goal.index(state[i])
            state_row, state_col = i // 3, i % 3
            goal_row, goal_col = goal_pos // 3, goal_pos % 3
            distance += abs(state_row - goal_row) + abs(state_col - goal_col)
    return distance

# Displaced Tiles Heuristic
def displaced_tiles_heuristic(state, goal):
    return sum(1 for i in range(9) if state[i] != goal[i] and state[i] != 0)

# A* Generic Search Function (Modified)
def a_star_generic(start, goal, heuristic_func=None, algorithm_type="A*"):

    start = tuple(map(int, start.flat))

    goal = tuple(map(int, goal.flat))
    open_set = []
    if algorithm_type == "DFS" or algorithm_type == "BFS":
        open_set = deque([(start, "")])  # For DFS/BFS, use a queue with state and path
    else:
        heapq.heappush(open_set, (0, start, ""))  # For A*, Dijkstra, use a priority queue

    closed_set = set()
    g_cost = {start: 0}  # g(n): cost to reach a state
    expanded_nodes = 0

    while open_set:
        if algorithm_type == "DFS":
            current_state, path = open_set.pop()
        else:
            if algorithm_type == "BFS":
                current_state, path = open_set.popleft()
            else:
                current_cost, current_state, path = heapq.heappop(open_set)
        
        if current_state == goal:
            return (list(path), expanded_nodes, g_cost[current_state] if algorithm_type != "BFS" else len(path))

        closed_set.add(current_state)
        expanded_nodes += 1
        
        for neighbor, move in get_neighbors(current_state):
            if neighbor in closed_set:
                continue
            
            # Calculate cost (g(n)) and heuristic (h(n))
            tentative_g_cost = g_cost[current_state] + 1  # Each move costs 1

            if algorithm_type == "A*" or algorithm_type == "Dijkstra":
                # For A* or Dijkstra, calculate the cost and heuristic
                f_cost = tentative_g_cost + (heuristic_func(neighbor, goal) if heuristic_func else 0)
            else:
                f_cost = tentative_g_cost  # For BFS/DFS, cost is only the number of moves

            # Check for Dijkstra or A* to update the g_cost
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                new_path = path + move
                if algorithm_type == "A*" or algorithm_type == "Dijkstra":
                    heapq.heappush(open_set, (f_cost, neighbor, new_path))
                else:
                    open_set.append((neighbor, new_path))

    return [], [], 0  # No solution found

def bfs(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int]:
    """
    Implement Breadth-First Search algorithm to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array.
                            Example: np.array([[1, 2, 3], [4, 0, 5], [6, 7, 8]])
                            where 0 represents the blank space
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array.
                          Example: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    
    Returns:
        Tuple[List[str], int]: A tuple containing:
            - List of moves to reach the goal state. Each move is represented as
              'U' (up), 'D' (down), 'L' (left), or 'R' (right), indicating how
              the blank space should move
            - Number of nodes expanded during the search

    Example return value:
        (['R', 'D', 'R'], 12) # Means blank moved right, down, right; 12 nodes were expanded
              
    """
    # TODO: Implement this function

    return a_star_generic(initial_state, goal_state, algorithm_type="BFS")
    # pass

def dfs(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int]:
    """
    Implement Depth-First Search algorithm to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
    """
    # TODO: Implement this function

    return a_star_generic(initial_state, goal_state, algorithm_type="DFS")
    # pass

def dijkstra(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement Dijkstra's algorithm to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration
            
    """
    # TODO: Implement this function

    return a_star_generic(initial_state, goal_state, algorithm_type="Dijkstra")
    # pass

def astar_dt(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement A* Search with Displaced Tiles heuristic to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration
              
    
    """
    # TODO: Implement this function

    return a_star_generic(initial_state, goal_state, heuristic_func=manhattan_heuristic, algorithm_type="A*")
    # pass

def astar_md(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement A* Search with Manhattan Distance heuristic to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration
    """
    # TODO: Implement this function

    return a_star_generic(initial_state, goal_state, heuristic_func=displaced_tiles_heuristic, algorithm_type="A*")
    # pass

# Example test case to help verify your implementation
if __name__ == "__main__":
    # Example puzzle configuration
    initial_state = np.array([
        [1, 2, 3],
        [4, 0, 5],
        [6, 7, 8]
    ])
    
    goal_state = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ])
    
    # Test each algorithm
    print("Testing BFS...")
    bfs_moves, bfs_expanded, bfs_cost = bfs(initial_state, goal_state)
    print(f"BFS Solution: {bfs_moves}")
    print(f"Nodes expanded: {bfs_expanded}")
    
    print("\nTesting DFS...")
    dfs_moves, dfs_expanded, dfs_cost = dfs(initial_state, goal_state)
    print(f"DFS Solution: {dfs_moves}")
    print(f"Nodes expanded: {dfs_expanded}")
    
    print("\nTesting Dijkstra...")
    dijkstra_moves, dijkstra_expanded, dijkstra_cost = dijkstra(initial_state, goal_state)
    print(f"Dijkstra Solution: {dijkstra_moves}")
    print(f"Nodes expanded: {dijkstra_expanded}")
    print(f"Total cost: {dijkstra_cost}")
    
    print("\nTesting A* with Displaced Tiles...")
    dt_moves, dt_expanded, dt_fscore = astar_dt(initial_state, goal_state)
    print(f"A* (DT) Solution: {dt_moves}")
    print(f"Nodes expanded: {dt_expanded}")
    print(f"Total cost: {dt_fscore}")
    
    print("\nTesting A* with Manhattan Distance...")
    md_moves, md_expanded, md_fscore = astar_md(initial_state, goal_state)
    print(f"A* (MD) Solution: {md_moves}")
    print(f"Nodes expanded: {md_expanded}")
    print(f"Total cost: {md_fscore}")



'''# Test Case 1: Solved Puzzle
puzzle_1 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 0]])

# Test Case 2: One move away from solved (move 8 to the bottom right)
puzzle_2 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 0, 8]])

# Test Case 3: Completely shuffled puzzle
puzzle_3 = np.array([[1, 2, 3],
                     [5, 8, 6],
                     [4, 7, 0]])

# Test Case 4: Another shuffled puzzle
puzzle_4 = np.array([[7, 2, 4],
                     [5, 0, 6],
                     [8, 3, 1]])

# Test Case 5: Reverse order
puzzle_5 = np.array([[8, 7, 6],
                     [5, 4, 3],
                     [2, 1, 0]])

# Test Case 6: Edge case - Blank in top-left corner
puzzle_6 = np.array([[0, 1, 2],
                     [3, 4, 5],
                     [6, 7, 8]])

# Test Case 7: Edge case - Blank in bottom-right corner
puzzle_7 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 0]])

# Test Case 8: Another scrambled but solvable puzzle
puzzle_8 = np.array([[1, 8, 2],
                     [0, 4, 3],
                     [7, 6, 5]])

# Test Case 9: Puzzle with multiple moves away from solution
puzzle_9 = np.array([[6, 5, 4],
                     [3, 2, 1],
                     [0, 8, 7]])

# Test Case 10: Unsolvable puzzle (odd number of inversions)
puzzle_10 = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [8, 7, 0]])'''

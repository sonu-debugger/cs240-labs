import heapq
import json
from typing import List, Tuple


def check_valid(
    state: list, max_missionaries: int, max_cannibals: int
) -> bool:  # 10 marks
    """
    Graded
    Check if a state is valid. State format: [m_left, c_left, boat_position].
    """

    m_left, c_left, boat_pos = state

    m_right = max_missionaries - m_left
    c_right = max_cannibals - c_left

    if (m_left < 0 or c_left < 0 or m_right < 0 or c_right < 0):
        return False
    
    if (m_left > 0 and c_left > m_left) or (m_right > 0 and c_right > m_right):
        return False
    
    return True

    # raise ValueError("check_valid not implemented")


def get_neighbours(
    state: list, max_missionaries: int, max_cannibals: int
) -> List[list]:  # 10 marks
    """
    Graded
    Generate all valid neighbouring states.
    """

    m_left, c_left, boat_pos = state

    m_right = max_missionaries - m_left
    c_right = max_cannibals - c_left

    neighbours = []

    moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

    for m_move, c_move in moves:
        if boat_pos == 1:
            new_state = [
                m_left - m_move, 
                c_left - c_move, 
                0
            ]
        else:
            new_state = [
                m_left + m_move, 
                c_left + c_move, 
                1
            ]

        if (m_move + c_move > 0 and m_move + c_move <= 2 and 
            check_valid(new_state, max_missionaries, max_cannibals)):
            neighbours.append(new_state)

    return neighbours

    # raise ValueError("get_neighbours not implemented")


def gstar(state: list, new_state: list) -> int:  # 5 marks
    """
    Graded
    The weight of the edge between state and new_state, this is the number of people on the boat.
    """

    m_old, c_old, boat_pos = state
    m_new, c_new, _ = new_state
    
    m_diff = abs(m_old - m_new)
    c_diff = abs(c_old - c_new)
    
    return m_diff + c_diff

    # raise ValueError("gstar not implemented")


def h1(state: list) -> int:  # 3 marks
    """
    Graded
    h1 is the number of people on the left bank.
    """

    m_left, c_left, boat_pos = state

    return m_left + c_left

    # raise ValueError("h1 not implemented")

def h2(state: list) -> int:  # 3 marks
    """
    Graded
    h2 is the number of missionaries on the left bank. 
    """

    m_left, c_left, boat_pos = state

    return m_left

    # raise ValueError("h2 not implemented")


def h3(state: list) -> int:  # 3 marks
    """
    Graded
    h3 is the number of cannibals on the left bank.
    """

    m_left, c_left, boat_pos = state

    return c_left

    # raise ValueError("h3 not implemented")


def h4(state: list) -> int:  # 3 marks
    """
    Graded
    Weights of missionaries is higher than cannibals.
    h4 = missionaries_left * 1.5 + cannibals_left
    """

    m_left, c_left, boat_pos = state

    return 1.5 * m_left + c_left

    # raise ValueError("h4 not implemented")


def h5(state: list) -> int:  # 3 marks
    """
    Graded
    Weights of missionaries is lower than cannibals.
    h5 = missionaries_left + cannibals_left*1.5
    """

    m_left, c_left, boat_pos = state

    return m_left + 1.5 * c_left

    # raise ValueError("h5 not implemented")

def astar_search(
    init_state: list, 
    final_state: list, 
    max_missionaries: int, 
    max_cannibals: int,
    heuristic_func
) -> Tuple[List[list], bool]:
    """
    Generic A* search implementation.
    """
    open_set = []
    heapq.heappush(open_set, (0, init_state))
    
    came_from = {}
    g_score = {tuple(init_state): 0}
    f_score = {tuple(init_state): heuristic_func(init_state)}
    
    monotone_satisfied = True
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        # Check monotone property
        if tuple(current) in f_score:
            for neighbor in get_neighbours(current, max_missionaries, max_cannibals):
                tentative_g_score = g_score.get(tuple(current), float('inf')) + gstar(current, neighbor)
                
                # Check monotone restriction
                if (f_score.get(tuple(current), float('inf')) > 
                    tentative_g_score + heuristic_func(neighbor)):
                    monotone_satisfied = False
        
        # Goal check
        if current == final_state:
            # Reconstruct path
            path = [current]
            while tuple(current) in came_from:
                current = came_from[tuple(current)]
                path.insert(0, current)
            return path, monotone_satisfied
        
        for neighbor in get_neighbours(current, max_missionaries, max_cannibals):
            tentative_g_score = g_score.get(tuple(current), float('inf')) + gstar(current, neighbor)
            
            if (tentative_g_score < g_score.get(tuple(neighbor), float('inf'))):
                came_from[tuple(neighbor)] = current
                g_score[tuple(neighbor)] = tentative_g_score
                f_score[tuple(neighbor)] = tentative_g_score + heuristic_func(neighbor)
                
                heapq.heappush(open_set, (f_score[tuple(neighbor)], neighbor))
    
    return [], monotone_satisfied

def astar_h1(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 28 marks
    """
    Graded
    Implement A* with h1 heuristic.
    This function must return path obtained and a boolean which says if the heuristic chosen satisfes Monotone restriction property while exploring or not.
    """

    return astar_search(init_state, final_state, max_missionaries, max_cannibals, h1)

    # raise ValueError("astar_h1 not implemented")


def astar_h2(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h2 heuristic.
    """

    return astar_search(init_state, final_state, max_missionaries, max_cannibals, h2)

    # raise ValueError("astar_h2 not implemented")


def astar_h3(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h3 heuristic.
    """

    return astar_search(init_state, final_state, max_missionaries, max_cannibals, h3)

    # raise ValueError("astar_h3 not implemented")

def astar_h4(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h4 heuristic.
    """

    return astar_search(init_state, final_state, max_missionaries, max_cannibals, h4)

    # raise ValueError("astar_h4 not implemented")


def astar_h5(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h5 heuristic.
    """

    return astar_search(init_state, final_state, max_missionaries, max_cannibals, h5)

    # raise ValueError("astar_h5 not implemented")


def print_solution(solution: List[list],max_mis,max_can):
    """
    Prints the solution path. 
    """
    if not solution:
        print("No solution exists for the given parameters.")
        return
        
    print("\nSolution found! Number of steps:", len(solution) - 1)
    print("\nLeft Bank" + " "*20 + "Right Bank")
    print("-" * 50)
    
    for state in solution:
        if state[-1]:
            boat_display = "(B) " + " "*15
        else:
            boat_display = " "*15 + "(B) "
            
        print(f"M: {state[0]}, C: {state[1]}  {boat_display}" 
              f"M: {max_mis-state[0]}, C: {max_can-state[1]}")


def print_mon(ism: bool):
    """
    Prints if the heuristic function is monotone or not.
    """
    if ism:
        print("-" * 10)
        print("|Monotone|")
        print("-" * 10)
    else:
        print("-" * 14)
        print("|Not Monotone|")
        print("-" * 14)


def main():
    try:
        testcases = [{"m": 3, "c": 3}]

        for case in testcases:
            max_missionaries = case["m"]
            max_cannibals = case["c"]
            
            init_state = [max_missionaries, max_cannibals, 1] #initial state 
            final_state = [0, 0, 0] # final state
            
            if not check_valid(init_state, max_missionaries, max_cannibals):
                print(f"Invalid initial state for case: {case}")
                continue
                
            path_h1,ism1 = astar_h1(init_state, final_state, max_missionaries, max_cannibals)
            path_h2,ism2 = astar_h2(init_state, final_state, max_missionaries, max_cannibals)
            path_h3,ism3 = astar_h3(init_state, final_state, max_missionaries, max_cannibals)
            path_h4,ism4 = astar_h4(init_state, final_state, max_missionaries, max_cannibals)
            path_h5,ism5 = astar_h5(init_state, final_state, max_missionaries, max_cannibals)
            print_solution(path_h1,max_missionaries,max_cannibals)
            print_mon(ism1)
            print("-"*50)
            print_solution(path_h2,max_missionaries,max_cannibals)
            print_mon(ism2)
            print("-"*50)
            print_solution(path_h3,max_missionaries,max_cannibals)
            print_mon(ism3)
            print("-"*50)
            print_solution(path_h4,max_missionaries,max_cannibals)
            print_mon(ism4)
            print("-"*50)
            print_solution(path_h5,max_missionaries,max_cannibals)
            print_mon(ism5)
            print("="*50)

    except KeyError as e:
        print(f"Missing required key in test case: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

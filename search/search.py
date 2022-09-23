# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack #First In Last Out，故使用Stack
    from game import Directions
 
    fringe = Stack() #存搜索树中的node
    finished = [] #记录展开过的state
 
    fringe.push((problem.getStartState(), [])) #item为（节点，到达该节点的一系列动作），压入首节点状态
    current = problem.getStartState()
    moves = []

    while problem.isGoalState(current) == False: #while循环搜索，只要当前节点不是目标则继续

        if current not in finished: #对没展开过的节点展开
            successors = problem.getSuccessors(current)
            finished += [current]
            for(state, direction, cost) in successors: #将展开的后继节点中未展开过的都压入fringe
                if state not in finished:
                    fringe.push((state, moves + [direction]))
        
        (current, moves) = fringe.pop() #pop最深的节点
    
    return moves #返回达成目标的动作系列

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue #First In First Out，故使用Queue，其余与DFS一样
    from game import Directions

    fringe = Queue()
    finished = []

    fringe.push((problem.getStartState(), []))
    current = problem.getStartState()
    moves = []

    while not problem.isGoalState(current):
        
        if current not in finished:
            successors = problem.getSuccessors(current)
            finished += [current]
            for(state, direction, cost) in successors:
                if state not in finished:
                    fringe.push((state, moves + [direction]))
        
        (current, moves) = fringe.pop() #pop最浅的节点
    
    return moves


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    from util import PriorityQueue #First In Least-cost Out，故使用Priority Queue
    from game import Directions

    fringe = PriorityQueue()
    finished = []

    fringe.push((problem.getStartState(), []), 0) #fringe存的信息增加了动作系列的代价来表征priority
    
    while not fringe.isEmpty():

        current, moves = fringe.pop() #pop代价最小的节点

        if problem.isGoalState(current):
            return moves

        if current not in finished:
            successors = problem.getSuccessors(current)
            finished += [current]
            for(state, direction, cost) in successors:
                if state not in finished:
                    fringe.push((state, moves + [direction]), problem.getCostOfActions(moves) + cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue #与UCS一样使用Priority Queue，只有代价计算方式改变了
    from game import Directions

    fringe = PriorityQueue()
    finished = []

    fringe.push((problem.getStartState(), []), heuristic(problem.getStartState(), problem)) #代价包含了节点启发函数的值，初始不为0
    
    while not fringe.isEmpty():

        current, moves = fringe.pop()

        if problem.isGoalState(current):
            return moves

        if current not in finished:
            successors = problem.getSuccessors(current)
            finished += [current]
            for(state, direction, cost) in successors:
                if state not in finished:
                    fringe.push((state, moves + [direction]), problem.getCostOfActions(moves) + cost + heuristic(state, problem)) #存入代价时除了原本的getCostOfActions之外还有启发函数值


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

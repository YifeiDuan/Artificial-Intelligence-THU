# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostDistances = [manhattanDistance(newPos, newGhostStates[index].getPosition()) for index in range(len(newGhostStates))] #计算与各ghost的曼哈顿距离作为评估依据
        nearestGhost = min(ghostDistances)  #最近幽灵
        if nearestGhost < 5: ghostScore = -(5 - nearestGhost)*5  #根据距最近幽灵的距离评估一个负值分数，距离越近该分数越低
        else: ghostScore = 0

        if len(newFood.asList()) > 0: foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()] #计算与各食物的曼哈顿距离
        else: foodDistances = [1]
        nearestFood = min(foodDistances)  #最近食物
        foodScore = 10/nearestFood  #根据最近食物的距离评估一个分数，距离越近该分数越高，且不会太高以至于抵消较近幽灵的负分
        
        totalScore = successorGameState.getScore() + ghostScore + foodScore  #最终评估分数包括本身getScore的结果和幽灵、食物两部分启发值
        return totalScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        currentAgent = 0
        currentDepth = 0
        minimaxResult = self.MiniMax(gameState,currentAgent, currentDepth)
        return minimaxResult[0]
    
    def MiniMax(self, gameState, currentAgent, currentDepth):

        if not currentAgent < gameState.getNumAgents():
            currentAgent = 0
            currentDepth += 1
        
        if currentDepth == self.depth: return self.evaluationFunction(gameState)

        if currentAgent == 0: return self.maxOperate(gameState, currentAgent, currentDepth)
        else: return self.minOperate(gameState, currentAgent, currentDepth)
    
    def maxOperate(self, gameState, currentAgent, currentDepth):

        if len(gameState.getLegalActions(currentAgent)) == 0:
            return self.evaluationFunction(gameState)
        
        maxValue = -float('inf')
        for action in gameState.getLegalActions(currentAgent):

            if action == "Stop" : continue
            
            getValue = self.MiniMax(gameState.generateSuccessor(currentAgent, action), currentAgent + 1, currentDepth)
            if type(getValue) == tuple: getValue = getValue[1]
            
            if getValue > maxValue:
                maxValue = getValue
                returnAction = action
        
        return (returnAction, maxValue)
    
    def minOperate(self, gameState, currentAgent, currentDepth):

        if len(gameState.getLegalActions(currentAgent)) == 0:
            return self.evaluationFunction(gameState)
        
        minValue = float('inf')
        for action in gameState.getLegalActions(currentAgent):

            if action == "Stop" : continue
            
            getValue = self.MiniMax(gameState.generateSuccessor(currentAgent, action), currentAgent + 1, currentDepth)
            if type(getValue) == tuple: getValue = getValue[1]
            
            if getValue < minValue:
                minValue = getValue
                returnAction = action
        
        return (returnAction, minValue)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        currentAgent = 0
        currentDepth = 0
        alpha = -float('inf')
        beta = float('inf')
        alphabetaResult = self.AlphaBeta(gameState,currentAgent, currentDepth, alpha, beta)
        return alphabetaResult[0]
    
    def AlphaBeta(self, gameState, currentAgent, currentDepth, alpha, beta):

        if not currentAgent < gameState.getNumAgents():
            currentAgent = 0
            currentDepth += 1
        
        if currentDepth == self.depth:
             return self.evaluationFunction(gameState)

        if currentAgent == 0: return self.maxOperate(gameState, currentAgent, currentDepth, alpha, beta)
        else: return self.minOperate(gameState, currentAgent, currentDepth, alpha, beta)
    
    def maxOperate(self, gameState, currentAgent, currentDepth, alpha, beta):
        
        if len(gameState.getLegalActions(currentAgent)) == 0:
            return self.evaluationFunction(gameState)
        
        maxValue = -float('inf')
        for action in gameState.getLegalActions(currentAgent):

            if action == "Stop" : continue
            
            getValue = self.AlphaBeta(gameState.generateSuccessor(currentAgent, action), currentAgent + 1, currentDepth, alpha, beta)
            if type(getValue) == tuple: getValue = getValue[1]
            
            if getValue > maxValue:
                maxValue = getValue
                returnAction = action
            
            if maxValue > beta: return(returnAction, maxValue) 
            
            alpha = max(maxValue, alpha)
        
        return (returnAction, maxValue)
    
    def minOperate(self, gameState, currentAgent, currentDepth, alpha, beta):
        
        if len(gameState.getLegalActions(currentAgent)) == 0:
            return self.evaluationFunction(gameState)
        
        minValue = float('inf')
        for action in gameState.getLegalActions(currentAgent):

            if action == "Stop" : continue
            
            getValue = self.AlphaBeta(gameState.generateSuccessor(currentAgent, action), currentAgent + 1, currentDepth, alpha, beta)
            if type(getValue) == tuple: getValue = getValue[1]
            
            if getValue < minValue:
                minValue = getValue
                returnAction = action
            
            if minValue < alpha: return (returnAction, minValue)
            
            beta = min(minValue, beta)
        
        return (returnAction, minValue)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        currentAgent = 0
        currentDepth = 0
        expectimaxResult = self.ExpectiMax(gameState,currentAgent, currentDepth)
        return expectimaxResult[0]
    
    def ExpectiMax(self, gameState, currentAgent, currentDepth):

        if not currentAgent < gameState.getNumAgents():
            currentAgent = 0
            currentDepth += 1
        
        if currentDepth == self.depth: return self.evaluationFunction(gameState)

        if currentAgent == 0: return self.maxOperate(gameState, currentAgent, currentDepth)
        else: return self.expectiOperate(gameState, currentAgent, currentDepth)
    
    def maxOperate(self, gameState, currentAgent, currentDepth):

        if len(gameState.getLegalActions(currentAgent)) == 0:
            return self.evaluationFunction(gameState)
        
        maxValue = -float('inf')
        for action in gameState.getLegalActions(currentAgent):

            if action == "Stop" : continue
            
            getValue = self.ExpectiMax(gameState.generateSuccessor(currentAgent, action), currentAgent + 1, currentDepth)
            if type(getValue) == tuple: getValue = getValue[1]
            
            if getValue > maxValue:
                maxValue = getValue
                returnAction = action
        
        return (returnAction, maxValue)
    
    def expectiOperate(self, gameState, currentAgent, currentDepth):

        if len(gameState.getLegalActions(currentAgent)) == 0:
            return self.evaluationFunction(gameState)
        
        expectiValue = 0
        probability = float(1/len(gameState.getLegalActions(currentAgent)))
        for action in gameState.getLegalActions(currentAgent):

            if action == "Stop" : continue
            
            getValue = self.ExpectiMax(gameState.generateSuccessor(currentAgent, action), currentAgent + 1, currentDepth)
            if type(getValue) == tuple: getValue = getValue[1]
            
            expectiValue += getValue * probability
        
        return expectiValue

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

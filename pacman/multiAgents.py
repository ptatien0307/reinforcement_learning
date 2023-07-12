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


from util import manhattanDistance, euclideanDistance
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
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance

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

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
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
        def minimax(state): 
            bestValue, bestAction = None, None
            for action in state.getLegalActions(0):
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = minimax(gameState)

        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)
        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))

        # return result        

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def alphabeta(state):
            # Init alpha and beta
            alpha = float("-Inf")
            beta = float("Inf")

            bestAction = None

            for action in state.getLegalActions(0):
                actionValue  = minValue(state.generateSuccessor(0, action), 1, 1, alpha, beta)
                if actionValue > alpha:
                    alpha = actionValue
                    bestAction = action
            return bestAction
        
        def minValue(state, agentIdx, depth, alpha, beta):
            # Agent is pacman
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1, alpha, beta)
            
            # If there's no legal actions, return current state score
            if (len(state.getLegalActions(agentIdx)) == 0):
                return self.evaluationFunction(state)
            
            beta0 = float("Inf")

            for action in state.getLegalActions(agentIdx):
                # Find and update beta
                actionValue = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
                beta0 = min(beta0, actionValue)

                # Prune branch
                if alpha >= beta0:
                    return beta0  

            return beta0


        def maxValue(state, agentIdx, depth, alpha, beta):
            # If current depth > foreseen depth or there's no legal actions, return current state score
            if (depth > self.depth) or (len(state.getLegalActions(0)) == 0):
                return self.evaluationFunction(state)
            
            alpha0 = float("-Inf")

            for action in state.getLegalActions(0):
                # Find and update alpha
                actionValue = minValue(state.generateSuccessor(agentIdx, action), 1, depth, alpha, beta)
                alpha0 = max(alpha0, actionValue)

                # Prune branch
                if alpha0 >= beta:
                    return alpha0  
                
            return alpha0

        action = alphabeta(gameState)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def expectimax(state):
            bestValue = float("-Inf")
            bestAction = None
            for action in state.getLegalActions(0):
                actionValue  = minValue(state.generateSuccessor(0, action), 1, 1)
                if actionValue > bestValue:
                    bestValue = actionValue
                    bestAction = action
            return bestAction

        def minValue(state, agentIdx, depth):
            # Agent is pacman
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            
            # If there's no legal actions, return current state score
            if (len(state.getLegalActions(agentIdx)) == 0):
                return self.evaluationFunction(state)
            
            # Compute average score
            sum = 0
            for action in state.getLegalActions(agentIdx):
                actionValue = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                sum += actionValue

            return sum / len(state.getLegalActions(agentIdx))


        def maxValue(state, agentIdx, depth):

            # If current depth > foreseen depth or there's no legal actions, return current state score
            if (depth > self.depth) or (len(state.getLegalActions(0)) == 0):
                return self.evaluationFunction(state)
            
            value = float("-Inf")

            for action in state.getLegalActions(0):
                actionValue = minValue(state.generateSuccessor(agentIdx, action), 1, depth)
                value = max(value, actionValue)
                
            return value
            

        action = expectimax(gameState)

        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

    if newCapsules:
        closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
    else:
        closestCapsule = 0

    if closestCapsule:
        closest_capsule = -3 / closestCapsule
    else:
        closest_capsule = 100

    if closestGhost:
        ghost_distance = -2 / closestGhost
    else:
        ghost_distance = -500

    foodList = newFood.asList()
    if foodList:
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    else:
        closestFood = 0

    return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule

def myEvaluationFunction(currentGameState):
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # Food
    foodLeft = len(newFood.asList())
    score += -15 * foodLeft

    capsuleLeft = len(newCapsules)
    score += -30 * capsuleLeft

    foodList = newFood.asList()


    # Ghost
    normalGhosts = list()
    scaredGhosts = list()
    
    for ghostState in newGhostStates:
        if ghostState.scaredTimer != 0:
            scaredGhosts.append(ghostState)
        else:
            normalGhosts.append(ghostState)

    if len(normalGhosts) != 0:
        closestNormalGhost = min([manhattanDistance(newPos, normal.getPosition()) for normal in normalGhosts])
        if closestNormalGhost != 0:
            score += -80 / closestNormalGhost
        else: 
            score += 0

    if len(scaredGhosts) != 0:
        closestScaredGhost = min([manhattanDistance(newPos, scared.getPosition()) for scared in scaredGhosts])
        score += -4* closestScaredGhost

    if foodLeft != 0:
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
        score += -3 * closestFood

    if len(scaredGhosts) == 0:
        if capsuleLeft != 0:
            closestCapsule = min([manhattanDistance(newPos, capsule) for capsule in newCapsules])
            score += -6 * closestCapsule


    return score
# Abbreviation
better = betterEvaluationFunction

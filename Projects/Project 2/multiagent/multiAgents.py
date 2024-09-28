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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        score = successorGameState.getScore()

        # Evaluate food distance
        foodList = newFood.asList()
        if len(foodList) > 0:
            minFoodDistance = util.manhattanDistance(newPos, foodList[0])
            for food in foodList[1:]:
                currentDistance = util.manhattanDistance(newPos, food)
                if currentDistance < minFoodDistance:
                    minFoodDistance = currentDistance
        else:
            minFoodDistance = 0

        # Reward food distance
        score += 1.0 / (minFoodDistance + 0.1)

        # Evaluate (un-scared & scared) ghost distance
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        ghostDistances = [util.manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions]
        scaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Penalize un-scared ghost distance
        for i in range(len(ghostDistances)):
            dist = ghostDistances[i]
            scaredTime = scaredTimes[i]
            if dist < 2 and scaredTime == 0:
                score -= 96

        # Reward scared ghost distance
        for dist, scaredTime in zip(ghostDistances, scaredTimes):
            if dist < 2 and scaredTime > 0:
                score += 69

        # return successorGameState.getScore()
        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        # Recursive Minimax Function
        def minimax(agentIndex, depth, gameState):

            # Base Cases
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman the Maximizing Player
            if agentIndex == 0:
                allActions = gameState.getLegalActions(0)

                firstAction = allActions[0]
                maxValue = minimax(1, depth, gameState.generateSuccessor(0, firstAction))

                for eachAction in allActions[1:]:
                    currentValue = minimax(1, depth, gameState.generateSuccessor(agentIndex, eachAction))
                    if currentValue > maxValue:
                        maxValue = currentValue

                return maxValue

            # Ghosts the Minimizing Player
            else:
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth += 1
                return min(minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, eachAction))
                           for eachAction in gameState.getLegalActions(agentIndex))


        # Run Minimax
        bestScore = float("-inf")
        bestAction = Directions.STOP
        for eachAction in gameState.getLegalActions(0):
            score = minimax(1, 0, gameState.generateSuccessor(0, eachAction))
            if score > bestScore:
                bestScore = score
                bestAction = eachAction
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        """
        Recursive Alpha Beta Function
        Alpha is the best value that the maximizing player can guarantee so far along the path to the root. 
        It starts as negative infinity and gets updated as the algorithm finds higher values (better moves for Pacman).
        
        Beta is the best value that the minimizing player can guarantee so far along the path to the root. 
        It starts as positive infinity and gets updated as the algorithm finds lower values (better moves for ghosts).
        """
        # Recursive Alpha Beta Function
        def alpha_beta(agentIndex, depth, gameState, alpha, beta):

            # Base Case
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman the Maximizing Player
            # Updates alpha and prunes branches where the value is greater than beta.
            if agentIndex == 0:
                maxValue = float("-inf")
                for eachAction in gameState.getLegalActions(agentIndex):
                    currentValue = alpha_beta(1, depth, gameState.generateSuccessor(agentIndex, eachAction), alpha,
                                              beta)
                    if currentValue > maxValue:
                        maxValue = currentValue

                    if maxValue > beta:
                        return maxValue
                    else:
                        alpha = max(alpha, maxValue)

                return maxValue

            # Ghosts the Minimizing Player
            # Updates beta and prunes branches where the value is less than alpha.
            else:
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth += 1
                minValue = float("inf")
                for eachAction in gameState.getLegalActions(agentIndex):
                    currentValue = alpha_beta(nextAgent, depth, gameState.generateSuccessor(agentIndex, eachAction),
                                              alpha, beta)

                    if currentValue < minValue:
                        minValue = currentValue

                    if minValue < alpha:
                        return minValue
                    else:
                        beta = min(beta, minValue)

                return minValue

        # Run Alpha Beta
        alpha = float("-inf")
        beta = float("inf")
        bestScore = float("-inf")
        bestAction = Directions.STOP
        for eachAction in gameState.getLegalActions(0):
            score = alpha_beta(1, 0, gameState.generateSuccessor(0, eachAction), alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = eachAction
            alpha = max(alpha, bestScore)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Recursive Expectimax Function
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman the Maximizing Player
            if agentIndex == 0:
                maxValue = float("-inf")

                for eachAction in gameState.getLegalActions(agentIndex):
                    currentValue = expectimax(1, depth, gameState.generateSuccessor(agentIndex, eachAction))
                    if currentValue > maxValue:
                        maxValue = currentValue

                return maxValue

            # Ghosts the Averaging Player
            else:
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth += 1
                actions = gameState.getLegalActions(agentIndex)
                return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, eachAction))
                           for eachAction in actions) / len(actions)

        # Run Expectimax
        bestScore = float("-inf")
        bestAction = Directions.STOP
        for eachAction in gameState.getLegalActions(0):
            score = expectimax(1, 0, gameState.generateSuccessor(0, eachAction))
            if score > bestScore:
                bestScore = score
                bestAction = eachAction
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION (generated by ChatGPT upon reading my code: This evaluation function for Pacman considers various game
    elements to calculate a score for any given     state. Wins and losses have extreme values to prioritize game
    outcomes. It adjusts scores based on the number of remaining food pellets and capsules, encouraging Pacman to
    consume them. The distance to ghosts affects the score differently depending on whether ghosts are scared, with
    closer scared ghosts increasing the score, and very close active ghosts heavily penalizing it. Finally, the distance
     to the nearest food pellet slightly penalizes the score, motivating Pacman to minimize this distance.
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    score -= 6.9 * len(foodPositions)
    if foodPositions:
        nearestFoodDistance = min(util.manhattanDistance(pacmanPosition, foodPos) for foodPos in foodPositions)
        score -= 0.69 * nearestFoodDistance

    capsulePositions = currentGameState.getCapsules()
    score -= 69 * len(capsulePositions)

    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    scaredTime = scaredTimes[0]

    for ghostState in ghostStates:
        distance = util.manhattanDistance(pacmanPosition, ghostState.getPosition())
        if ghostState.scaredTimer > 0:
            score += 6.9 * (scaredTime / distance)
        else:
            if distance < 2:
                score -= 690 / distance


    return score

# Abbreviation
better = betterEvaluationFunction

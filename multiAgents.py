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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        import sys
        old_food = currentGameState.getFood()

        total_score = 0.0
        for x in xrange(old_food.width):
            for y in xrange(old_food.height):
                if(old_food[x][y]):
                    d = manhattanDistance((x,y),newPos)
                    if (d==0):
                        total_score +=100
                    else:
                        total_score += 1.0/(d*d)
                        
        #total_score = successorGameState.getScore()
        for ghost in newGhostStates:
            d=manhattanDistance(ghost.getPosition(), newPos)
            if (d<=1):
                if(ghost.scaredTimer!=0):
                    total_score += 2000
                else:
                    total_score -=200
       
        # for capsule in currentGameState.getCapsules()

        return total_score

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
        """
        "*** YOUR CODE HERE ***"
        import sys
        def result(gameState, agent, action):
            return gameState.generateSuccessor(agent, action)

        def utility(gameState):
            return self.evaluationFunction(gameState)

        def terminalTest(gameState, depth):
            return depth == 0 or gameState.isWin() or gameState.isLose()

        def max_value(gameState, agent, depth):
            if terminalTest(gameState, depth): return utility(gameState)
            v = -sys.maxint
            for a in gameState.getLegalActions(agent):
                v = max(v,min_value(result(gameState,agent,a),1,depth))
            return v

        def min_value(gameState, agent, depth):
            if terminalTest(gameState, depth): return utility(gameState)
            v = sys.maxint
            for a in gameState.getLegalActions(agent):
                if (agent == gameState.getNumAgents()-1):
                    v = min(v, max_value(result(gameState,agent,a), 0, depth-1))
                else:
                    v = min(v, min_value(result(gameState,agent,a), agent+1, depth))
            return v

        v = -sys.maxint
        actions = []
        for a in gameState.getLegalActions(0):        
            u = min_value(result(gameState, 0, a), 1, self.depth)
            if u == v: actions.append(a)
            elif u>=v: v = u; actions = [a]     

        return random.choice(actions)
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import sys

        def result(gameState, agent, action):
            return gameState.generateSuccessor(agent, action)

        def utility(gameState):
            return self.evaluationFunction(gameState)

        def terminalTest(gameState, depth):
            return depth == 0 or gameState.isWin() or gameState.isLose()

        def max_value(gameState, alpha, beta, agent, depth):
            if terminalTest(gameState, depth): return utility(gameState)
            v = -sys.maxint
            for a in gameState.getLegalActions(agent):
                v = max(v,min_value(result(gameState,agent,a), alpha, beta, 1,depth))
                # Cut branch
                if v > beta: return v
                alpha = max(alpha, v)
            return v

        def min_value(gameState, alpha, beta, agent, depth):
           # If it is a leaf we calculate the value of this (evaluationFunction)
            if terminalTest(gameState, depth): return utility(gameState)
            v = sys.maxint
            for a in gameState.getLegalActions(agent):
                if (agent == gameState.getNumAgents()-1):
                    v = min(v, max_value(result(gameState,agent,a), alpha, beta, 0, depth-1))
                else:
                    v = min(v, min_value(result(gameState,agent,a), alpha, beta, agent+1, depth))
                # Cut branch
                if v < alpha: 
                  return v
                beta = min(beta, v)
            return v

        # AlphaBetaSearch
        v = -sys.maxint 
        actions = []
        # Alpha =  Most favorable value for a maximum
        # beta = Most favorable value for a minimum
        alpha = -sys.maxint

        # The Pacman takes the minimum from each branch and obtains the maximum (alpha).
        for a in gameState.getLegalActions(0):       
            u = min_value(result(gameState, 0, a), alpha, sys.maxint, 1, self.depth)
            if u == v: 
              actions.append(a)
            elif u>=v: 
              v = u 
              actions = [a]     
            alpha = max(alpha, v)   

        return random.choice(actions)
        # util.raiseNotDefined()

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
        import sys
        def result(gameState, agent, action):
            return gameState.generateSuccessor(agent, action)

        def utility(gameState):
            return self.evaluationFunction(gameState)

        def terminalTest(gameState, depth):
            return depth == 0 or gameState.isWin() or gameState.isLose()

        def max_value(gameState, agent, depth):
            if terminalTest(gameState, depth): return utility(gameState)
            v = -sys.maxint
            for a in gameState.getLegalActions(agent):
                v = max(v, min_value(result(gameState, agent, a), 1, depth))
            return v

        def min_value(gameState, agent, depth):
            if terminalTest(gameState, depth): return utility(gameState)
            v = sys.maxint
            listValues = []
            for a in gameState.getLegalActions(agent):
                if (agent == gameState.getNumAgents() - 1):
                    listValues.append(min(v, max_value(result(gameState, agent, a), 0, depth - 1)))
                else:
                    listValues.append(min(v, min_value(result(gameState, agent, a), agent + 1, depth)))
                    
            average = sum(listValues)/len(gameState.getLegalActions(agent))
            return average

        v = -sys.maxint
        actions = []
        for a in gameState.getLegalActions(0):        
            u = min_value(result(gameState, 0, a), 1, self.depth)
            if u == v: actions.append(a)
            elif u >= v: v = u; actions = [a] 

        return random.choice(actions)
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  import sys

  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  old_food = currentGameState.getFood()
  total_score = 0.0

  for x in xrange(old_food.width):
    for y in xrange(old_food.height):
      if(old_food[x][y]):
        d = manhattanDistance((x,y),newPos)
        if (d==0):
          total_score +=100
        else:
          total_score += 1.0/(d*d)
                  
  #total_score = successorGameState.getScore()
  for ghost in newGhostStates:
    d=manhattanDistance(ghost.getPosition(), newPos)
    if (d<=1):
      if(ghost.scaredTimer!=0):
        total_score += 2000
      else:
        total_score -=200

  for capsule in currentGameState.getCapsules():
    d = manhattanDistance(capsule, newPos) # posicion del pacman  
    if(d == 0):
      total_score += 200            # cuando llega a distancia 0 se la come
    else:
      total_score += 1.0/(d*d)      # lleva al pacman a por la comida           
  
  return total_score + 90 * currentGameState.getScore()  # para que el pacman intente siempre hacer el maximo score

# Abbreviation
better = betterEvaluationFunction

#util.raiseNotDefined()


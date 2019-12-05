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
					d = manhattanDistance((x, y), newPos)
					if (d == 0):
						total_score += 100
					else:
						total_score += 1.0 / (d * d)
						
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
				v = max(v,min_value(result(gameState, agent, a), 1, depth))
			return v

		def min_value(gameState, agent, depth):
			if terminalTest(gameState, depth): return utility(gameState)
			v = sys.maxint
			for a in gameState.getLegalActions(agent):
				if (agent == gameState.getNumAgents() - 1):
					v = min(v, max_value(result(gameState,agent,a), 0, depth-1))
				else:
					v = min(v, min_value(result(gameState, agent, a), agent + 1, depth))
			return v

		v = -sys.maxint
		actions = []
		for a in gameState.getLegalActions(0):        
			u = min_value(result(gameState, 0, a), 1, self.depth)
			if u == v: actions.append(a)
			elif u >= v: v = u; actions = [a]     

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
				v = max(v,min_value(result(gameState, agent, a), alpha, beta, 1, depth))
				if v > beta: return v
				alpha = max(alpha, v)
			return v

		def min_value(gameState, alpha, beta, agent, depth):
			if terminalTest(gameState, depth): return utility(gameState)
			v = sys.maxint
			for a in gameState.getLegalActions(agent):
				if (agent == gameState.getNumAgents() - 1):
					v = min(v, max_value(result(gameState,agent,a), alpha, beta, 0, depth - 1))
				else:
					v = min(v, min_value(result(gameState, agent, a), alpha, beta, agent + 1, depth))
				if v < alpha: 
				  return v
				beta = min(beta, v)
			return v

		# AlphaBetaSearch
		v = -sys.maxint 
		actions = []
		alpha = -sys.maxint

		for a in gameState.getLegalActions(0):       
			u = min_value(result(gameState, 0, a), alpha, sys.maxint, 1, self.depth)
			if u == v: 
			  actions.append(a)
			elif u >= v: 
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

		  Here are some method calls that might be useful when implementing minimax.

		  gameState.getLegalActions(agentIndex):
			Returns a list of legal actions for an agent
			agentIndex=0 means Pacman, ghosts are >= 1

		  All ghosts should be modeled as choosing uniformly at random from their
			legal moves.

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
				v = max(v, min_value(result(gameState, agent, a), 1, depth))
		
			return v

		def min_value(gameState, agent, depth):
			if terminalTest(gameState, depth): return utility(gameState)
			v = sys.maxint	

			expected_vals = []			
			for a in gameState.getLegalActions(agent):
				if (agent == gameState.getNumAgents() - 1):
					expected_vals.append(min(v, max_value(result(gameState, agent, a), 0, depth - 1)))
				else:
					expected_vals.append(min(v, min_value(result(gameState, agent, a), agent + 1, depth)))			
		
			return sum(expected_vals) / len(gameState.getLegalActions(agent))

		v = -sys.maxint
		actions = []

		for a in gameState.getLegalActions(0):        
			u = min_value(result(gameState, 0, a), 1, self.depth)
			if u == v: actions.append(a)
			elif u >= v: v = u; actions = [a]

		return random.choice(actions)

"""

Write a better evaluation function for pacman in the provided function betterEvaluationFunction. The evaluation function should evaluate states, rather than actions like your reflex agent evaluation function did. You may use any tools at your disposal for evaluation, including your search code from the last project. With depth 2 search, your evaluation function should clear the smallClassic layout with one random ghost more than half the time and still run at a reasonable rate (to get full credit, Pacman should be averaging around 1000 points when he's winning).
Grading: the autograder will run your agent on the smallClassic layout 10 times. We will assign points to your evaluation function in the following way:

If you win at least once without timing out the autograder, you receive 1 points. Any agent not satisfying these criteria will receive 0 points.
+1 for winning at least 5 times, +2 for winning all 10 times
+1 for an average score of at least 500, +2 for an average score of at least 1000 (including scores on lost games)
+1 if your games take on average less than 30 seconds on the autograder machine. The autograder is run on EC2, so this machine will have a fair amount of resources, but your personal computer could be far less performant (netbooks) or far more performant (gaming rigs).
The additional points for average score and computation time will only be awarded if you win at least 5 times.

"""
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """

    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    pacmanPosition = currentGameState.getPacmanPosition()
    activeGhosts = [] # Keep active ghosts(can eat pacman)
    scaredGhosts = [] # Keep scared ghosts(pacman should eat them for extra points)
    
    totalCapsules = len(currentGameState.getCapsules()) # Keep total capsules
    totalFood = len(food) # Keep total remaining food
    
    myEval = 0 # Evaluation value

    # Fix active and scared ghosts #
    for ghost in ghosts:
        if ghost.scaredTimer: # Is scared ghost
            scaredGhosts.append(ghost)
        else:
            activeGhosts.append(ghost)

    # Score weight: 1.5                                    #
    # Better score -> better result                        #
    # Score tell us some informations about current state  #
    # but the weight is low. We don't care enough for this #
    # But we do not want to lose                           #
    myEval += 1.5 * currentGameState.getScore()

    # Food weight: -10                                   #
    # Pacman will receive 10 points if he eats one food  #
    # If our state has a lot of food this is very bad    #
    # We are far away from our goal. If pacman eats food #
    # Evaluation value will be better in the new         #
    # state, because remaining food is less              #
    myEval += -10 * totalFood

    # Capsules weight: -20                            #
    # Same like food but pacman gains a huge amount   #
    # of points if he eats ghosts. So our goal is to  #
    # eat a capsule and then eat a ghost              #
    # For that reason pacman should eat capsules more #
    # frequently than food                            #
    # Weight food < Weight capsules                   #
    myEval += -20 * totalCapsules

    # Keep distances from food, active and scared ghosts #
    foodDistances = []
    activeGhostsDistances = []
    scaredGhostsDistances = []

    # Find distances #
    for item in food:
        foodDistances.append(manhattanDistance(pacmanPosition, item))

    for item in activeGhosts:
        scaredGhostsDistances.append(manhattanDistance(pacmanPosition, item.getPosition()))

    for item in scaredGhosts:
        scaredGhostsDistances.append(manhattanDistance(pacmanPosition, item.getPosition()))

    # Fix evaluation based on food distances  #
    # It is very bad for pacman to have close #
    # food. He must eat it.                   #
    # Close food weight: -1                   #
    # Quite close food weight: -0.5           #
    # Far away food weight: -0.2              #
    for item in foodDistances:
        if item < 3:
            myEval += -1 * item
        if item < 7:
            myEval += -0.5 * item
        else:
            myEval += -0.2 * item

    # Fix evaluation based on scared ghosts distances    #
    # It is very bad for pacman to have close scared     #
    # ghosts. He must eat them so as to gain many points #
    # We should prefer to eat a ghost rather than eat a  #
    # close food                                         #
    # Close scared ghosts weight: -20                    #
    # Quite close scared ghosts weight: -10              #
    for item in scaredGhostsDistances:
        if item < 3:
            myEval += -20 * item
        else:
            myEval += -10 * item

    # Fix evaluation base on active ghosts distances    #
    # Pacman should avoid active ghosts                 #
    # Close ghost weight: 3                             #
    # Quite close ghost weight: 2                       #
    # Far away ghosts weight: 0.5                       #
    # We should prefer ghosts remaining far away        #
    for item in activeGhostsDistances:
        if item < 3:
            myEval += 3 * item
        elif item < 7:
            myEval += 2 * item
        else:
            myEval += 0.5 * item

    return myEval

# Abbreviation
better = betterEvaluationFunction

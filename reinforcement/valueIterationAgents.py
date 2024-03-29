# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        for _ in range(self.iterations):
          values = util.Counter() # Copy the value
          for state in self.mdp.getStates(): # For each state in the mdp
            QValueForAction = util.Counter() # Copy the V(s)
            for action in self.mdp.getPossibleActions(state): # Compute Q(s,a)
              QValueForAction[action] = self.computeQValueFromValues(state, action) # Compute Q values for each action
            values[state] = QValueForAction[QValueForAction.argMax()] # Which is the action that has de greatest Q value
            
          self.values = values  

        # At this point we have generated a policy map in N iterations        

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        QValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
          # Q(s,a) = Sumatori de T(s, a, s') * [R(s, a, s') + BETA*V(s')]
          QValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])

        return QValue

    # Computes the optimal policy:  pi*(s) = optimal action from state s
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        # Has the enviornment entered a terminal state? This means there are no successors
        if self.mdp.isTerminal(state): return None

        # get the best possible action for the state
        QValueForAction = util.Counter()

        for action in self.mdp.getPossibleActions(state):
          # how good is an action = q-value (which considers all possible outcomes)
          QValueForAction[action] = self.computeQValueFromValues(state, action)

        # return the best action, e.g. 'north'
        return QValueForAction.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

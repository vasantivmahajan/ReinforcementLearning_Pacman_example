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
        self.values = util.Counter()

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #start at iteration zero
        iteration_count=0
        
        #perform value iteration
        while iteration_count < self.iterations:
            
            value_iter = util.Counter()
            #get all the states
            allStates=mdp.getStates()
            
            #iterate over the states 
            for state in allStates:
                #if the state is a non terminal state perform the following operations
                if not mdp.isTerminal(state):
                    
                    vals = util.Counter()
                    #get all possible actions for the state
                    allActions = mdp.getPossibleActions(state)
                    #Call the Qvalue function and fetch the Q value for state and action pair
                    for action in allActions:
                        vals[action] = self.computeQValueFromValues(state, action)
                    #set the Q value for the state to the maximum Q value for each state action pairs
                    value_iter[state] = max(vals.values())
            iteration_count += 1
            #update the policy with the best Qvalue
            self.values = value_iter.copy()  

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
        "*** YOUR CODE HERE ***"
        #get the Transition function and nextStates
        #stateProbPairs=self.mdp.getTransitionStatesAndProbs(state,action)
        total=0
        #iterate over probabilities (transition functions) and next states
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
          total += prob * (self.mdp.getReward(state,action,nextState) + (self.discount*self.values[nextState]))
        return total
                        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #if the state is terminal state return None
        if self.mdp.isTerminal(state):
            return None
        
        #get all possible actions for the state
        actions=self.mdp.getPossibleActions(state)
        
        #if there aren't any actions in that state return None
        if (len(actions) == 0):
            return None
        
        values = util.Counter()
        #for all the actions compute the Qvalue
        for action in actions:
            values[action] = self.computeQValueFromValues(state, action)
        #calculate the maximum Qvalue and return it
        return values.argMax()
    
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

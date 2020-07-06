# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from util import nearestPoint
import game
MAX_DEPTH = 5

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveMinimaxAgent', second='DefensiveMinimaxAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class CoreAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    TEAM = 0
    if self.index % 2 == 1:
      self.TEAM = 1

    actions = gameState.getLegalActions(self.index)
    actions.remove('Stop')


    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    best_value = float('-inf')
    for action in actions:
      if TEAM == 0:
        value = self.red_minimax(gameState.generateSuccessor(self.index, action), 1, self.index + 1)
      else:
        value = self.red_minimax(gameState.generateSuccessor(self.index, action), 1, self.index + 1)
      if value > best_value:
        best_action = action
        best_value = value
    print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return best_action


  def red_minimax(self, gameState, depth, agent):
    # if agent == gameState.getNumAgents():
    #   agent = 0
    #   depth = depth + 1

    if gameState.isOver() or depth == MAX_DEPTH:
      return self.evaluate(gameState)

    if agent in gameState.getRedTeamIndices():
      best_value = float('-inf')
      actions = gameState.getLegalActions(agent)
      for action in actions:
        value = self.red_minimax(gameState.generateSuccessor(agent, action), depth+1, agent+1)
        if value > best_value:
          best_value = value
      return best_value
    else:
      best_value = float('inf')
      actions = gameState.getLegalActions(agent)
      for action in actions:
        value = self.red_minimax(gameState.generateSuccessor(agent, action), depth+1, agent-1)
        if value < best_value:
          best_value = value
      return best_value

  def blue_minimax(self, gameState, depth, agent):
    if agent == gameState.getNumAgents():
      agent = 0
      depth = depth + 1

    if gameState.isOver() or depth == MAX_DEPTH:
      return self.evaluate(gameState)

    if agent in gameState.getBlueTeamIndices():
      best_value = float('-inf')
      actions = gameState.getLegalActions(agent)
      for action in actions:
        value = self.blue_minimax(gameState.generateSuccessor(agent, action), depth, agent+1)
        if value > best_value:
          best_value = value
      return best_value
    else:
      best_value = float('inf')
      actions = gameState.getLegalActions(agent)
      for action in actions:
        value = self.blue_minimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
        if value < best_value:
          best_value = value
      return best_value

  #
  # def getSuccessor(self, gameState, action):
  #   """
  #   Finds the next successor which is a grid position (location tuple).
  #   """
  #   successor = gameState.generateSuccessor(self.index, action)
  #   pos = successor.getAgentState(self.index).getPosition()
  #   if pos != nearestPoint(pos):
  #     # Only half a grid position was covered
  #     return successor.generateSuccessor(self.index, action)
  #   else:
  #     return successor


  def evaluate(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights(gameState)
    score = 100*gameState.getScore() + features*weights
    print (score)
    return score

  def getFeatures(self, gameState):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    features['successorScore'] = self.getScore(gameState)
    return features

  def getWeights(self, gameState):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}


class OffensiveMinimaxAgent(CoreAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def getFeatures(self, gameState):
    features = util.Counter()
    foodList = self.getFood(gameState).asList()
    features['successorScore'] = -len(foodList)  # self.getScore(successor)
    features['ghostNear'] = 0

    # Compute distance to the nearest food

    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
      myPos = gameState.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    myState = gameState.getAgentState(self.index)
    for enemy in gameState.getBlueTeamIndices():
      enemyState = gameState.getAgentState(enemy)
      enemyPos = enemyState.getPosition()
      if (not enemyState.isPacman) and myState.isPacman and (self.getMazeDistance(myPos, enemyPos)<5):
        features['ghostNear'] = 1

    return features

  def getWeights(self, gameState):
    return {'successorScore': 100, 'distanceToFood': -1, 'ghostNear': -1000}


class DefensiveMinimaxAgent(CoreAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState):
    features = util.Counter()

    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    foodList = self.getFood(gameState).asList()
    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
      myPos = gameState.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    # if action == Directions.STOP: features['stop'] = 1
    # rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    # if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState):
    return {'numInvaders': -1000, 'onDefense': 10, 'invaderDistance': -10, 'stop': -100,  'distanceToFood':-1}


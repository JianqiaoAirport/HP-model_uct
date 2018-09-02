import numpy as np
import copy
from operator import itemgetter
import random
import logging

import config

def rollout_policy_fn(state):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly

    legal_actions = state.legal_action_list
    legal_actions_int = []
    for action in legal_actions:
        legal_actions_int.append(config.ACTION_INT_DICT[action])

    action_probs = np.random.rand(len(legal_actions))
    action_probs = (1 - 0.99) * action_probs + 0.01 * np.random.dirichlet(0.2 * np.ones(len(legal_actions)))
    return zip(legal_actions_int, action_probs)


def policy_value_fn(state):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(state.legal_action_list))/len(state.legal_action_list)

    legal_actions = state.legal_action_list
    legal_actions_int = []
    for action in legal_actions:
        legal_actions_int.append(config.ACTION_INT_DICT[action])

    return zip(legal_actions_int, action_probs), 0

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # a map from action to TreeNode
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prior_p
        self.squared_Q_from_subtrees = 0
        self.sp = 0

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self.n_visits += 1
        # Update Q, a running average of values for all visits.
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits
        self.squared_Q_from_subtrees += leaf_value ** 2

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self.u = (c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        self.sp = np.sqrt((abs(self.squared_Q_from_subtrees - self.n_visits * self.Q ** 2)+40) / (self.n_visits + 1))
        return self.Q + self.u + self.sp + 0.00001 * random.random()

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        value = 0

        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_action(config.INT_ACTION_DICT[action])

        action_probs, leaf_value = self._policy(state)
        # Check for end of game
        end = state.is_terminal()
        score = -state.get_energy()
        if not end:
            node.expand(action_probs)
            # Update value and visit count of nodes in this traversal.
            leaf_value = self._evaluate_rollout(state)
            node.update_recursive(leaf_value)
            return value
        else:
            leaf_value = score
            value = score

            if score > - config.BEST_ENERGY:
                config.BEST_ENERGY = -score
                config.BEST_ACTION_LIST = state.action_list

                print("In play_out: ")
                print(config.BEST_ENERGY)
                print(config.BEST_ACTION_LIST)
                print("--------------")

                logging.info("In play_out: ")
                logging.info(config.BEST_ENERGY)
                logging.info(config.BEST_ACTION_LIST)
                logging.info("--------------")


            node.update_recursive(leaf_value)
            return value

    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        for i in range(limit):
            end = state.is_terminal()
            score = -state.get_energy()
            if end:
                if score > - config.BEST_ENERGY:
                    config.BEST_ENERGY = -score
                    config.BEST_ACTION_LIST = state.action_list
                    print("In roll_out: ")
                    print(config.BEST_ENERGY)
                    print(config.BEST_ACTION_LIST)
                    print("--------------")

                    logging.info("In roll_out: ")
                    logging.info(config.BEST_ENERGY)
                    logging.info(config.BEST_ACTION_LIST)
                    logging.info("--------------")
                return score
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_action(config.INT_ACTION_DICT[max_action])
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")

    def get_move_probs(self, state, temp=1e-3):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        Return: the selected action
        """
        highest_value = 0
        for n in range(self._n_playout + 1):
            state_copy = copy.deepcopy(state)
            value = self._playout(state_copy)
            if highest_value < value:
                highest_value = value

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.n_visits) for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))  # for training policy-value network

        return acts, act_probs, highest_value

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root.parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class UCTAgent(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=4000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_agent_ind(self, p):
        self.player = p

    def reset_agent(self):
        self.mcts.update_with_move(-1)

    def get_action(self, state, temp=1e-3, return_prob=False):
        """

        :param state:  HP-evn instance
        :param temp:  temprature
        :param return_prob: MCTS visit cont
        :return: move, move_probs, max_value
        """
        sensible_moves = state.legal_action_list
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(len(config.ACTION_VECTOR_DICT.keys())-1)
        if len(sensible_moves) > 0:
            acts, probs, value = self.mcts.get_move_probs(state, temp=temp)
            move_probs[list(acts)] = probs
            #  if the mcts result doesn't point to an action strongly, give other actions an opportunity
            if np.max(move_probs) < 0.85:
                temp = 0.7
            probs_with_temp = softmax(1.0 / temp * np.log(probs + 1e-10))
            if True:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=(1-config.EPSILON)*probs_with_temp + config.EPSILON*np.random.dirichlet(config.DIRICHLET*np.ones(len(probs_with_temp)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs_with_temp)
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs, value
            else:
                return move
        else:
            print("WARNING: no legal actions")

    def __str__(self):
        return "MCTS {}".format(self.player)

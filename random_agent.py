import numpy as np
import random

import config
import global_variables


class RandomAgent:
    """Agent based on MCTS"""

    def __init__(self, is_self_play=True):
        self._is_selfplay = is_self_play
        pass


    def get_action(self, state, temp=1e-3, return_prob=False):
        sensible_moves = state.legal_action_list
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(len(global_variables.ACTION_VECTOR_DICT.keys()) - 1)
        if len(sensible_moves) > 0:
            probs = np.random.dirichlet(np.ones(2*config.DIMENSIONALITY), size=1)
            acts = np.argmax(probs)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=(1-config.EPSILON)*probs + config.EPSILON*np.random.dirichlet(config.DIRICHLET*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: no legal actions")

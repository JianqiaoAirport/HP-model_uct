import time
import logging

import HP_env
import config


class SearchLogic:
    def __init__(self, hp_seq, agent):
        self.env = HP_env.HPEnv(hp_seq)
        self.agent = agent
        self.search_record = []

    def search(self):
        action_lists = []
        prob_lists = []
        energy = 0

        for i in range(len(self.env.hp_seq)-1):
            action_list = self.env.action_list
            action_lists.append(action_list)
            action, prob_list = self.agent.get_action(self.env, return_prob=True)
            self.env.do_action(config.INT_ACTION_DICT[action])
            prob_lists.append(prob_list)

            if self.env.is_terminal():
                energy = self.env.get_energy()
                print("Energy: "+str(energy))
                logging.info("Energy: "+str(energy))
                print("Action list: "+str(self.env.action_list))
                logging.info("Action list: "+str(self.env.action_list))
                break

        # (hp_seq+action_list) means state
        return (self.env.hp_seq, action_lists, prob_lists, energy)

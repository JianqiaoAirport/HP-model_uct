from __future__ import print_function
import random
import numpy as np
from collections import deque
import logging
import os

from HP_env import HPEnv
# from mcts_agent import MCTSAgent
from uct_agent import UCTAgent
import config


class TrainPipeline:
    def __init__(self, init_model_path=None):
        self.env = HPEnv(hp_seq=config.HP_SEQ)
        # training params
        self.learn_rate = config.LEARNING_RATE
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = config.TEMP  # the temperature param
        self.n_playout = config.N_PLAYOUT  # num of simulations for each move
        self.c_puct = config.C_PUCT
        self.buffer_size = config.BUFFER_SIZE
        self.batch_size = config.BATCH_SIZE  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = config.EPOCH  # num of train_steps for each update
        self.kl_targ = config.KL_TARG
        self.check_freq = 50
        self.game_batch_num = config.GAME_BATCH_NUM
        # self.best_energy = 0.0
        # self.best_action_list = []
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.mcts_player = UCTAgent(c_puct=self.c_puct, n_playout=config.N_PLAYOUT)

    # def get_equi_data(self, play_data):
    #     """augment the data set by rotation and flipping
    #     play_data: [(state, mcts_prob, winner_z), ..., ...]
    #     """
    #     extend_data = []
    #     for state, mcts_porb, winner in play_data:
    #         for i in [1, 2, 3, 4]:
    #             # rotate counterclockwise
    #             equi_state = np.array([np.rot90(s, i) for s in state])
    #             equi_mcts_prob = np.rot90(np.flipud(
    #                 mcts_porb.reshape(self.board_height, self.board_width)), i)
    #             extend_data.append((equi_state,
    #                                 np.flipud(equi_mcts_prob).flatten(),
    #                                 winner))
    #             # flip horizontally
    #             equi_state = np.array([np.fliplr(s) for s in equi_state])
    #             equi_mcts_prob = np.fliplr(equi_mcts_prob)
    #             extend_data.append((equi_state,
    #                                 np.flipud(equi_mcts_prob).flatten(),
    #                                 winner))
    #     return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""

        ave_energy = 0
        for i in range(n_games):
            terminal_energy, action_list, play_data = self.env.generate_training_data(self.mcts_player, temp=self.temp)
            if terminal_energy < config.BEST_ENERGY:
                config.BEST_ENERGY = terminal_energy
                config.BEST_ACTION_LIST = action_list
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            # play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
            ave_energy += terminal_energy

            if i % 5 == 4:
                ave_energy /= 5
                logging.info("Average energy:{:.3f}".format(ave_energy))
                print("Average energy:{:.3f}".format(ave_energy))
                ave_energy = 0
            return ave_energy


    def run(self):
        """run the training pipeline"""

        try:
            average_energy = 0
            for i in range(self.game_batch_num):
                print("-----"+str(i)+"-----")
                e = -self.collect_selfplay_data(self.play_batch_size)
                average_energy += e
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                logging.info("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if i % 10 == 0:
                    print("Best energy: "+str(config.BEST_ENERGY))
                    logging.info("Best energy: "+str(config.BEST_ENERGY))
                    print("Best action list: "+str(config.BEST_ACTION_LIST))
                    logging.info("Best action list: "+str(config.BEST_ACTION_LIST))
                    average_energy /= 10
                    logging.info("Average energy:{:.3f}".format(average_energy))
                    print("Average energy:{:.3f}".format(average_energy))
                    average_energy = 0

                # if i % 300 == 13:
                #     self.policy_value_net.save_model(i)
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':

    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    if not os.path.exists(config.WORK_PATH + 'logs'):
        os.makedirs(config.WORK_PATH + 'logs')

    logging.basicConfig(filename=config.WORK_PATH + 'logs/'+'training_record_'+config.MODEL_NAME+'.log', filemode="a", level=logging.DEBUG)

    # training_pipeline = TrainPipeline(init_model_path=config.WORK_PATH + "models/BiLSTM_Attention_24_2018-08-20_21_38_17/model-1813.ckpt")
    training_pipeline = TrainPipeline()
    training_pipeline.run()

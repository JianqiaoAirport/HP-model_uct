from __future__ import print_function
import random
import numpy as np
from collections import deque
import logging
import os

from HP_env import HPEnv
from uct_agent import UCTAgent
from p_v_network import ResNet
import config
import global_variables


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

        if init_model_path:
            # start training from an initial policy-value net
            self.policy_value_net.restore_model(init_model_path)
        else:
            # start training from a new policy-value net
            self.policy_value_net = ResNet()


    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""

        ave_energy = 0
        for i in range(n_games):
            terminal_energy, action_list, play_data = self.env.generate_training_data(self.mcts_player, temp=self.temp)
            if terminal_energy < global_variables.BEST_ENERGY:
                global_variables.BEST_ENERGY = terminal_energy
                global_variables.BEST_ACTION_LIST = action_list
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

    def policy_update(self, step):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0][0] for data in mini_batch])
        mcts_probs_batch = [data[1] for data in mini_batch]
        energy_batch = np.array([data[2] for data in mini_batch])[:, np.newaxis]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        if step % 20 == 19:
            logging.info(("number of 1 in batch:{:.1f}, "
                          "number of 2 in batch:{:.1f}, "
                          "number of 3 in batch:{:.1f}, "
                          "number of 4 in batch:{:.1f}, "
                          "number of 5 in batch:{:.1f}, "
                          "number of 6 in batch:{:.1f}, "
                          "number of 7 in batch:{:.1f}, "
                          "number of 8 in batch:{:.1f}, "
                          "number of 9 in batch:{:.1f}, "
                          ).format(np.sum(energy_batch == 1),
                                   np.sum(energy_batch == 2),
                                   np.sum(energy_batch == 3),
                                   np.sum(energy_batch == 4),
                                   np.sum(energy_batch == 5),
                                   np.sum(energy_batch == 6),
                                   np.sum(energy_batch == 7),
                                   np.sum(energy_batch == 8),
                                   np.sum(energy_batch == 9),))

        for i in range(self.epochs):
            loss, policy_loss, value_loss, value_out_mean, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    energy_batch,
                    self.learn_rate*self.lr_multiplier, step*self.epochs+i)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # explained_var_old = (1 -
        #                      np.var(np.array(winner_batch) - old_v.flatten()) /
        #                      np.var(np.array(winner_batch)))
        # explained_var_new = (1 -
        #                      np.var(np.array(winner_batch) - new_v.flatten()) /
        #                      np.var(np.array(winner_batch)))
        print(("kl:{:.5f}, "
               "lr_multiplier:{:.3f}, "
               "loss:{:.3f}, "
               "policy loss:{:.3f}, "
               "value loss:{:.3f}, "
               "value out mean:{:.3f}, "
               "entropy:{:.3f}, "
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        policy_loss,
                        value_loss,
                        value_out_mean,
                        entropy,
                        ))
        logging.info(("kl:{:.5f}, "
               "lr_multiplier:{:.3f}, "
               "loss:{:.3f}, "
               "policy loss:{:.3f}, "
               "value loss:{:.3f}, "
               "value out mean:{:.3f}, "
               "entropy:{:.3f}, "
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        policy_loss,
                        value_loss,
                        value_out_mean,
                        entropy,
                        ))
        #  adjust c_puct to a reasonable number
        self.mcts_player.mcts._c_puct = value_out_mean
        return loss, entropy


    def run(self):
        """run the training pipeline"""

        try:
            average_energy = 0
            for i in range(self.game_batch_num):
                print("-----"+str(i)+"-----")
                e = -self.collect_selfplay_data(self.play_batch_size)

                if len(self.data_buffer) > self.batch_size:
                    for j in range(config.N_POLICY_UPDATE):
                        loss, entropy = self.policy_update(i*config.N_POLICY_UPDATE+j)

                average_energy += e

                if i % 10 == 0:
                    print("Best energy: " + str(global_variables.BEST_ENERGY))
                    logging.info("Best energy: " + str(global_variables.BEST_ENERGY))
                    print("Best action list: " + str(global_variables.BEST_ACTION_LIST))
                    logging.info("Best action list: " + str(global_variables.BEST_ACTION_LIST))
                    average_energy /= 10
                    logging.info("Average energy:{:.3f}".format(average_energy))
                    print("Average energy:{:.3f}".format(average_energy))
                    average_energy = 0

                if i % 15 == 13:
                    self.policy_value_net.save_model(i)
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_WHEN_TRAINING

    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    if not os.path.exists(config.WORK_PATH + 'logs'):
        os.makedirs(config.WORK_PATH + 'logs')

    logging.basicConfig(filename=config.WORK_PATH + 'logs/'+'training_record_'+config.MODEL_NAME+'.log', filemode="a", level=logging.DEBUG)

    # training_pipeline = TrainPipeline(init_model_path=config.WORK_PATH + "models/BiLSTM_Attention_24_2018-08-20_21_38_17/model-1813.ckpt")
    training_pipeline = TrainPipeline()
    training_pipeline.run()

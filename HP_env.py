import logging
import sys
import numpy as np

import config


class HPEnv:
    def __init__(self, hp_seq, action_list=["START_TOKEN"], dimensionality=config.DIMENSIONALITY):
        self.hp_seq = hp_seq
        self.action_list = action_list
        self.dimensionality = dimensionality
        self.history_location, self.pointer = self.get_history_location(hp_seq, action_list)
        self.legal_action_list = self.get_legal_action_list()
        self.encoded_state = self.generate_encoded_state()

    def get_history_location(self, hp_seq, action_list):
        """
        history_location: dict {(x1, y1): 1, (x2, y2): -1, ...}
        :param hp_seq: HP sequence (HHHP: [1,1,1, -1])
        :param action_list: ([UP, RIGHT, DOWN, LEFT, (FORWARD, BACKWARD)])
        :return: history_location and pointer to current coordinate
        """
        history_location = {}
        pointer = config.ACTION_VECTOR_DICT["START_TOKEN"]
        for i, action in enumerate(action_list):
            if action == "START_TOKEN":
                if i != 0:
                    print("Start token not at the beginning error")
                    logging.error("Start token not at the beginning error")
                    sys.exit(1)
                    #  else:
                history_location[pointer] = hp_seq[i]
            else:
                if action in config.ACTION_VECTOR_DICT.keys():
                    pointer = self.tuple_add(pointer, config.ACTION_VECTOR_DICT[action])
                else:
                    print("Undefined action error")
                    logging.error("Undefined action error")
                    sys.exit(1)
                if pointer in history_location.keys():
                    print("Collision error")
                    logging.error("Collision error")
                    sys.exit(1)
                else:
                    history_location[pointer] = hp_seq[i]

        return history_location, pointer

    def action_is_legal(self, action):
        """
        :param action: action
        :return: ?the action is legal
        """
        if action in self.legal_action_list:
            return True
        else:
            return True

    def do_action(self, action):
        """
        execute action in the environment
        :param action: chosen action
        :return: ?the action is legal
        """
        if self.action_is_legal(action):
            self.action_list.append(action)
            self.pointer = self.tuple_add(self.pointer, config.ACTION_VECTOR_DICT[action])
            self.history_location[self.pointer] = self.hp_seq[len(self.action_list)-1]
            self.legal_action_list = self.get_legal_action_list()
            self.encoded_state = self.generate_encoded_state()
            return True
        else:
            logging.warning("Illegal action: " + str(action))
            print("Illegal action: " + str(action))
            return False

    def get_energy(self):
        history_location = self.history_location
        action_list = self.action_list  # {(x1, y1): 1, (x2, y2): -1, ...}
        if len(action_list) != len(self.hp_seq):
            return 1  # collision happens, return 1

        if len(history_location.keys()) != len(action_list):
            return "Not match error"
        if action_list[0] != "START_TOKEN":
            return "No start token error"

        continuous_h_count = 0  # record the number of continuous H residues
        neighbor_h_count = 0  # record the number of neighbor H residues
        pointer = config.ACTION_VECTOR_DICT["START_TOKEN"]  # pointer_y to the current residues in np array
        previous_residue = None  # previous_residues

        for i, action in enumerate(action_list):
            if action == "START_TOKEN":
                if i != 0:
                    return "Start token not at the beginning error"
                #  else:
                current_residue = history_location[pointer]
            else:
                #  construct the 2D or 3D structure
                if action in config.ACTION_VECTOR_DICT.keys():
                    pointer = self.tuple_add(pointer, config.ACTION_VECTOR_DICT[action])
                else:
                    return "Undefined action error"

                if pointer not in history_location.keys():
                    return "Unseen location error"
                else:
                    current_residue = history_location[pointer]

            if current_residue == -1:
                #  count the number of continuous_h_count and neighbor_h_count
                if current_residue == previous_residue:
                    continuous_h_count += 1

                #  count the number of neighbor_h_count and neighbor_h_count
                actions = list(config.ACTION_VECTOR_DICT.keys())
                actions.remove("START_TOKEN")
                for a in actions:
                    #  get the nearby points, to see whether they are H residues
                    temp_pointer = self.tuple_add(pointer, config.ACTION_VECTOR_DICT[a])
                    if temp_pointer in history_location.keys():
                        if history_location[temp_pointer] == -1:
                            neighbor_h_count += 1

            previous_residue = current_residue
        # calculate the energy
        energy = continuous_h_count - neighbor_h_count / 2

        return energy

    def get_legal_action_list(self):
        """
        :return: current legal action list
        """
        visited_list = self.history_location.keys()
        legal_action_list = []

        actions = list(config.ACTION_VECTOR_DICT.keys())
        actions.remove("START_TOKEN")

        for action in actions:
            new_place = self.tuple_add(self.pointer, config.ACTION_VECTOR_DICT[action])
            if new_place not in visited_list:
                legal_action_list.append(action)
        return legal_action_list

    def is_terminal(self):
        """
        :return: ?current state is terminal
        """
        if not self.legal_action_list:
            return True
        elif len(self.action_list) == len(self.hp_seq):
            return True
        else:
            return False

    def tuple_add(self, a, b):
        """
        coordinate + vector
        :param a: coordinate
        :param b: action vector (ex. right = (1, 0))
        :return: new_coordinate = a + b
        """
        c = list(range(len(a)))
        for i in range(len(a)):
            c[i] = a[i] + b[i]
        return tuple(c)

    def reset_env(self):
        self.action_list = ["START_TOKEN"]
        self.history_location, self.pointer = self.get_history_location(self.hp_seq, self.action_list)
        self.legal_action_list = self.get_legal_action_list()
        self.encoded_state = self.generate_encoded_state()

    def generate_encoded_state(self):
        """
        generate input state suitable for neural network
        :return: encoded state
        """
        arr = []
        for i, residue in enumerate(self.hp_seq):
            if i < len(self.action_list):
                arr.append([residue])
                arr[i].extend(list(config.ACTION_VECTOR_DICT[self.action_list[i]]))
            else:
                arr.append([residue])
                arr[i].extend([0 for _ in range(config.DIMENSIONALITY)])
        return np.array(arr)[np.newaxis, :, :]

    def generate_training_data(self, agent, temp=1e-3):
        """
        :param agent: agent
        :param temp: temperature
        :return: energy_list: best energy for each state encountered during mcts, if mcts doesn't reach terminal energy,
        the energy is replaced by the terminal energy
        """
        self.reset_env()
        agent.reset_agent()

        encoded_states = []
        prob_lists = []
        terminal_value = 0
        value_list = []

        for i in range(len(self.hp_seq)-1):
            state = self.encoded_state
            encoded_states.append(state)
            action, prob_list, value = agent.get_action(self, temp=temp, return_prob=True)
            self.do_action(config.INT_ACTION_DICT[action])
            prob_lists.append(prob_list)
            value_list.append(value)

            if self.is_terminal():
                terminal_value = - self.get_energy()
                print("Energy: " + str(-terminal_value))
                logging.info("Energy: " + str(-terminal_value))
                print("Action list: " + str(self.action_list))
                logging.info("Action list: " + str(self.action_list))
                break
        for i, e in enumerate(value_list):
            for j in range(i):
                if value_list[j] < value_list[i]:
                    value_list[j] = value_list[i]

        # (hp_seq+action_list) means state
        return terminal_value, self.action_list, zip(encoded_states, prob_lists, value_list)


#  test
if __name__ == "__main__":
    hp_seq = [-1, -1, -1, 1, -1, 1, 1, -1, -1]
    # action_list = ["START_TOKEN", "DOWN", "DOWN", "RIGHT", "UP", "RIGHT", "UP", "LEFT", "FORWARD"]

    hp_env = HPEnv(hp_seq)

    hp_env.do_action("DOWN")

    print(hp_env.get_energy())


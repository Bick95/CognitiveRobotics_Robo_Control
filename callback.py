import os

def callback(locals_, globals_):
    """
    Method passed to RL (PPO2) agent in order to make it save checkpoints during training.
    Implemented by letting agent save its internal NN model at a given frequency (=checkpoint_frequency).
    :param locals_: Local variables inside RL agent
    :param globals_: Global variables iside python interpreter
    :return:
    """
    self_ = locals_['self']
    updates_ = locals_['update']

    if updates_ % self_.checkpoint_frequency == 0:
        if not os.path.exists(self_.check_point_location):
            os.makedirs(self_.check_point_location)

        check_point = updates_  # int(updates_ / self_.checkpoint_frequency)
        self_.save(self_.check_point_location + "checkpoint_" + str(check_point))

    return True

import os

def callback(locals_, globals_):
    """
        Method passed to RL (PPO2) agent in order to make it save checkpoints during training.
        Implemented by letting agent save its internal NN model at a given frequency (=checkpoint_frequency).
        :param locals_: Local variables inside RL agent
        :param globals_: Global variables iside python interpreter
        :return: -
    """
    self_ = locals_['self']
    updates_ = locals_['update']

    if updates_ % self_.checkpoint_frequency == 0:
        if not os.path.exists(self_.check_point_location):
            os.makedirs(self_.check_point_location)

        check_point = updates_  # int(updates_ / self_.checkpoint_frequency)
        self_.save(self_.check_point_location + "checkpoint_" + str(check_point))

    # TODO: save training progress to file: create file in main? -- Specify row headlines:
    #  Update_nr | grasp | avg_graps_time_steps | std_graps_time_steps | min_graps_time_steps | max_graps_time_steps
    if updates_ % self_.training_eval_frequency == 0:
        with open('training_eval.csv', 'a') as fd:

            # TODO URGENT : reset counters IFF(!!!) this method is called...
            #  Rows:
            #  Update_nr | grasp | avg_gr_time_steps | std_gr_time_steps | min_gr_time_steps | max_gr_time_steps

            row = str("...")  # get data...
            fd.write(row)

            self_.get_env().reset_training_eval_counters()

    return True

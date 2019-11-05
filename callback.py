import os, csv
import statistics

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

    # Save checkpoint
    if updates_ % self_.checkpoint_frequency == 0:
        if not os.path.exists(self_.path):
            os.makedirs(self_.path)

        check_point = updates_
        self_.save(self_.path + "checkpoint_" + str(check_point))

    # Log training progress
    if updates_ % self_.log_train_progress_frequency == 0:
        if not os.path.exists(self_.path):
            os.makedirs(self_.path)
        '''
            Logging:
                ['Update_nr',                # Current count of weight updates performed so far
                 'Grasps',                   # Nr of grasps over last X weight updates
                 'Avg_grasp_time_steps',     # Average of time steps needed in a simulation to get from init
                                             # pose to attaining goal, averaged over the time steps recorded
                                             # for all successful grasps over last X weight updates
                 'Std_grasp_time_steps',
                 'Max_grasp_time_steps',
                 'Min_grasp_time_steps',
                 'Total_time_steps'                # Total time steps simulated so far
                 ]
        '''
        # TODO: make it work for multiple environments in vectorized environment; Possibly 1 log file per env
        grasps = self_.get_env().envs[0].grasps_per_update_interval
        grasp_times = self_.get_env().envs[0].grasp_time_steps_needed_per_update_interval
        #if len(grasp_times) < 1:
        #    grasp_times.extend([float('nan')]*(1-len(grasp_times)))
        if len(grasp_times) == 0:
            # Mean() takes only lists of at least one element
            grasp_times.extend([float('nan')])

        avg_grasp_time_steps = statistics.mean(grasp_times)

        if len(grasp_times) == 1:
            # Std() takes only lists of at least two elements
            std_grasp_time_steps = float('nan')
        else:
            std_grasp_time_steps = statistics.stdev(grasp_times)

        max_grasp_time_steps = max(grasp_times)
        min_grasp_time_steps = min(grasp_times)
        total_time_steps = self_.num_timesteps

        row = [updates_,
               grasps,
               avg_grasp_time_steps,
               std_grasp_time_steps,
               max_grasp_time_steps,
               min_grasp_time_steps,
               total_time_steps]

        with open(self_.path + "training_eval.csv", "a") as fp:
            wr = csv.writer(fp, dialect='excel', quoting=csv.QUOTE_ALL)
            wr.writerow(row)

        self_.get_env().envs[0].reset_logged_train_data()

    return True

import os, sys
import json, csv
import numpy as np
import time

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gym
from stable_baselines import PPO2
from customRobotEnv import PandaRobotEnv
from stable_baselines.common.vec_env import DummyVecEnv

def create_dir(direct):
    """
        Ensure that a given path exists.
        :param direct: Directory to be created when necessary.
        :return: -
    """
    if not os.path.exists(direct):
        os.makedirs(direct)

parentDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_READ = os.path.join(parentDirectory, 'Results', 'PPO2')
PATH_READ = '../Results/PPO2/'
PATH_WRITE = "../FinalEvaluation/"
print(PATH_READ)


def get_complete_trials(path):
    """
        Takes a path and checks which of the folders/archives inside given folder (specified by path) contain a
        final model generated during the training progress, and hence determines which data sets in terms of archives
        are complete (due to complete training process).
    :param path: Path to archive(s)-location
    :return:    dirs: List of directories/archives only containing those with complete data sets.
                list_outtakes: List of data directories/archives containing not completed test runs indicated by
                               absence of final model.
    """
    dirs = os.listdir(path)
    list_outtakes = []
    print('All dirs:')
    print(dirs)
    print()

    for direct in dirs:
        f = []
        for (dirpath, dirnames, filenames) in os.walk(path+direct):
            f.extend(filenames)
            break
        if 'final_model.zip' in f:
            pass
        else:
            print('Incomplete data at: ' + direct)
            list_outtakes.append(direct)
            dirs.remove(direct)

    return dirs, list_outtakes

def return_parameter_specification_id_for_file(file_name):
    """
        Given a folder name of a folder containing data created during model's training, the method returns which
        parameter specification file was used to specify the parameters as a function of which the agent's training
        proceeded.
    :param file_name: Folder containing data created during training.
    :return: -
    """
    with open(file_name) as json_file:
        data = json.load(json_file)
    try:
        # Data might not contain requested element
        return data['provided_params_file']
    except KeyError:
        return None

def remove_redundant_runs(path, data_directories):
    """
        Filter out redundant back-up-test runs which are too many.
        :param path: Path to directories potentially to be included
        :param data_directories: Candidate directories to potentially be included in analysis
        :return:
            During training, each train run was called with a given parameter specification provided in a named file.
            This function reads out from the data created during training which parameter specification file was
            used for generating the respective data set specified by a given data_directory.
    """
    parameter_specification_ids = dict()
    discarded_data_directories = []
    maxElements = 5
    #print('Potential data directories: ')
    #print(data_directories)
    for directory in data_directories:
        #print('To be checked: ' + path+directory+'/params.json')
        parameter_specification_id = return_parameter_specification_id_for_file(path+directory+'/params.json')
        if parameter_specification_id is not None:
            if parameter_specification_id not in parameter_specification_ids.keys():
                id_list = [directory]
                parameter_specification_ids[parameter_specification_id] = id_list
            else:
                prev_records = parameter_specification_ids[parameter_specification_id]
                if len(prev_records) < maxElements:
                    prev_records.append(directory)
                    parameter_specification_ids[parameter_specification_id] = prev_records
                else:
                    discarded_data_directories.append(directory)
        else:
            discarded_data_directories.append(directory)

    return parameter_specification_ids, discarded_data_directories


def save_which_data_was_used(direct, used_dict, not_used_lists):
    """
        Specify which folders containing data were used for evaluation. Each folder contains all data generated during
        a single training run.
    :param direct: Path to a folder containing all used/non-used folders/directories.
    :param used_dict: Dictionary containing the list of folders used for evaluation per param_setting_specification_id
    :param not_used_lists: Two list containing folders excluded from evaluation;
                           1. Excluded due to being incomplete
                           2. Excluded due to number of data directories to be included per param setting being exceeded
    :return: -
    """
    create_dir(direct)

    print('Used:')
    print(used_dict)
    print('Not used:')
    print(not_used_lists)

    # Save as csv which data was used (nicer to read)
    with open(direct + "/" + "used_data.csv", "w") as f:
        w = csv.writer(f, dialect='excel', quoting=csv.QUOTE_ALL)
        for key, val_list in used_dict.items():
            for val in val_list:
                w.writerow([key, val])
    f.close()

    # Save as csv which data was not used (nicer to read)
    with open(direct + "/" + "not_used_data.csv", "w") as f:
        w = csv.writer(f, dialect='excel', quoting=csv.QUOTE_ALL)
        w.writerow(['Incomplete data sets:'])
        w.writerow([''])
        for val in not_used_lists[0]:
            w.writerow([val])
        for _ in range(2):
            w.writerow([''])
        w.writerow(['Excluded due to number of data directories to be included per param setting being exceeded:'])
        w.writerow([''])
        for val in not_used_lists[1]:
            w.writerow([val])
        w.writerow([''])
        w.writerow(['-- Done.'])

    f.close()


def save_dict_to_file(direct, name, data_dict):
    """
        Dictionary saving.
    :param direct: Folder where to store emission file
    :param name: Indication how to call resulting file
    :param data_dict: Dictionary containing parameter-setting-ids and their respective evaluation-statistics
    :return: - (save to file)
    """
    create_dir(direct)

    # Save as csv
    with open(direct + "/" + name + ".csv", "w") as f:
        w = csv.writer(f, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
        for param_id, statistic in data_dict.items():
            w.writerow([param_id, statistic])

    f.close()


def load_model_params(params_path):
    """
        Returns the dictionary of parameters used for originally training a model.
    :param params_path: Path to folder containing all data, including param settings, associated with a given trained
                        agent
    :return: Dictionary containing the parameter specifications used to originally train a given agent
    """
    with open(params_path) as json_file:
        params = json.load(json_file)

    return params


def test_run(model_path, params_path):

    # Run simulation 100 times for a single model

    num_test_runs = 2   # FIXME
    iterations = 10     # FIXME

    print('Running 100 tests on model: ' + model_path)
    print('Using params: ' + params_path)

    # Load params
    params = load_model_params(params_path)
    print(params)

    # FIXME: remove again
    params['render'] = True

    # Creating test env
    env = PandaRobotEnv(renders=params['render'],
                        fixedActionRepetitions=params['fixed_action_repetitions'],
                        distSpecifications=params['dist_specification'],
                        maxDist=params['maxDist'],
                        maxDeviation=params['maxDeviation'],
                        maxSteps=iterations,
                        evalFlag=True)
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run, hence vectorize

    # Load model
    model = PPO2.load(model_path)

    # Run simulation
    test_scores = []  # Over 100 eval runs
    test_times_avgs = []
    test_times_stds = []
    for _ in range(num_test_runs):
        score_over_test_run = 0
        previous_time_steps = time_steps = 0
        reaching_times_over_test_run = []
        obs = env.reset()

        while time_steps < iterations:

            env.envs[0].set_step_counter(time_steps)
            action, _states = model.predict(obs)
            obs, _, _, info = env.step(action)  # Assumption: eval conducted on single env only!

            reward, time_steps, done = info[0][:]

            print(info)

            time.sleep(0.1)


            if done:
                print('Accumulated rew.:' + str(score_over_test_run))
                print('Time: ' + str(time_steps))

                if reward > 0:
                    score_over_test_run += reward
                    time_steps_reaching = time_steps - previous_time_steps
                    reaching_times_over_test_run.append(time_steps_reaching)
                    print('Time to goal: ' + str(time_steps_reaching))
                previous_time_steps = time_steps

                print(score_over_test_run)
                print(reaching_times_over_test_run)

                obs = env.reset()
        test_scores.append(score_over_test_run)  # Test score obtained per test run
        test_times_avgs.append(np.nanmean(np.array(reaching_times_over_test_run)))  # Average reaching time per test run
        test_times_stds.append(np.nanstd(np.array(reaching_times_over_test_run)))  # Std's of reaching time per test run

    # Return mean test score for model
    print('Scores:')
    print(test_scores)
    print('Times:')
    print(test_times_avgs)

    # Return:
    # mean test score over number of test runs for a single model,
    # mean over 100*[mean time per test run],
    # mean over 100*[std of mean time per test run]
    return np.nanmean(np.array(test_scores)), np.nanmean(np.array(test_times_avgs)), np.nanmean(np.array(test_times_stds))


def evaluate_measurements_per_param_specification(path, params):
    print('Params used and associated test runs:')
    eval_scores_per_param_setting = dict()
    mean_time_per_param_setting = dict()
    std_time_per_param_setting = dict()
    print(params)
    print()
    # Iterate through all parameter settings on which models were trained
    for param_specification_id, model_folder_lst in params.items():
        print(param_specification_id)
        avg_score_per_model = []
        avg_time_per_model = []
        std_time_per_model = []
        # Iterate through all models trained on a single parameter setting
        for model_folder in model_folder_lst:
            model_path = path + model_folder + '/final_model.zip'
            params_path = path + model_folder + '/params.json'
            print('Going to evaluate:' + model_folder)
            # Evaluate each model 100 times and take the average score obtained per test run,
            #                                        the average time needed to get to the goal per test run, and
            #                                        the standard deviation of average time needed to get to the goal
            #                                         per test run.
            #                                         All measures averaged oder the 100 test runs per model:
            model_avg_score, avg_time_steps, avg_std_time_steps = test_run(model_path=model_path, params_path=params_path)  # Run 100 test runs per model
            # Collect measurements for all models trained on given parameter setting
            avg_score_per_model.append(model_avg_score)
            avg_time_per_model.append(avg_time_steps)
            std_time_per_model.append(avg_std_time_steps)

            print('Avg Score: ' + str(model_avg_score))
            print('Avg time:' + str(avg_time_steps))
            print('Avg std of time steps needed: ' + str(avg_std_time_steps))

        print('Scores obtained: ')
        print(np.array(avg_score_per_model))
        # Calculate averages per parameter setting/specification over all models trained on that parameter specification
        eval_scores_per_param_setting[param_specification_id] = np.nanmean(np.array(avg_score_per_model))
        mean_time_per_param_setting[param_specification_id] = np.nanmean(np.array(avg_time_per_model))
        std_time_per_param_setting[param_specification_id] = np.nanmean(np.array(std_time_per_model))
        print()

    print('Param-specification-Avg-scores:')
    print(eval_scores_per_param_setting)
    print(mean_time_per_param_setting)
    print(std_time_per_param_setting)
    print()

    return eval_scores_per_param_setting, mean_time_per_param_setting, std_time_per_param_setting


candidate_dirs, list_outtakes_failure = get_complete_trials(PATH_READ)
# list_outtakes_failure == incomplete data sets

used_dict, filtered_out = remove_redundant_runs(PATH_READ, candidate_dirs)
# filtered_out == directories rejected due to max number of directories to include exceeded

# used_dict == key : parameter-specification-id ; value : list of directories (including data of a single test run)
#                                                         associated with a given parameter specification given by key

not_used_lists = [list_outtakes_failure + filtered_out]

eval_scores_per_param_setting, mean_time_per_param_setting, std_time_per_param_setting = \
    evaluate_measurements_per_param_specification(PATH_READ, used_dict)

# Save to file which data was included in final analysis and which wasn't
save_which_data_was_used(PATH_WRITE+"EvaluatedData", used_dict, not_used_lists)

save_dict_to_file(direct=PATH_WRITE+'Statistics', name='param_average_scores', data_dict=eval_scores_per_param_setting)
save_dict_to_file(direct=PATH_WRITE+'Statistics', name='param_average_time', data_dict=mean_time_per_param_setting)
save_dict_to_file(direct=PATH_WRITE+'Statistics', name='param_average_std_time', data_dict=std_time_per_param_setting)



print('Complete and sufficient runs:')
print(candidate_dirs)
print('Outtakes:')
print(not_used_dict)
print()
print('Used parameter id\'s and the test runs that used them:')
print(used_dict)


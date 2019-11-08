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


def clean_parameter_specification_id_string(param_id):
    # Clean id which was itself a directory+id beforehand
    param_id = param_id.replace('ParameterSettings/', '')
    param_id = param_id.replace('.json', '')
    param_id = param_id.replace('.', '_')
    param_id = param_id.replace('/', '_')
    return param_id


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

    # Save as csv which data was used (nicer to read)
    with open(direct + "/" + "used_data.csv", "w") as f:
        w = csv.writer(f, dialect='excel', quoting=csv.QUOTE_ALL)

        # Save overview: for param-specification-id: [param-specification-id, count-of-used-models]
        w.writerow(['-- Overview about how many models per parameter-specification were used for evaluation:'])
        w.writerow(['Parameter specification id:', 'Count of models:'])
        for key, val_list in used_dict.items():
            key = clean_parameter_specification_id_string(key)
            w.writerow([key, len(val_list)])
        w.writerow(['-- End of list.'])
        w.writerow([''])

        # Save actual data pairs [param-specification-id, used-data-set-containing-trained-model]
        w.writerow(['-- Which models were used per parameter-specification:'])
        w.writerow(['Parameter specification id:', 'Model:'])
        for key, val_list in used_dict.items():
            key = clean_parameter_specification_id_string(key)
            for val in val_list:
                w.writerow([key, val])
        w.writerow(['-- End of list.'])
    f.close()

    # Save as csv which data was not used (nicer to read)
    with open(direct + "/" + "not_used_data.csv", "w") as f:
        w = csv.writer(f, dialect='excel', quoting=csv.QUOTE_ALL)
        w.writerow(['-- Incomplete data sets:'])
        for val in not_used_lists[0]:
            w.writerow([val])
        w.writerow(['-- End of list.'])
        for _ in range(2):
            w.writerow([''])
        w.writerow(['-- Excluded due to number of data directories to be included per param setting being exceeded:'])
        for val in not_used_lists[1]:
            w.writerow([val])
        w.writerow(['-- End of list.'])

    f.close()


def save_mean_and_std_time_to_file(direct, name, data_dict_mean, data_dict_std):
    """
        Write both average mean grasping time steps per parameter-specification-id and the corresponding std to file,
        both in consecutive columns.
    :param direct: Folder where to store emission file
    :param name: Indication how to call resulting file
    :param data_dict_mean: Dict containing mean grasping time steps per parameter-specification-id (saved row-wise)
    :param data_dict_std: Dict containing mean std of mean grasping time steps per parameter-specification-id
    :return: -
    """
    create_dir(direct)

    # Save as csv
    with open(direct + "/" + name + ".csv", "w") as f:
        w = csv.writer(f, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(['Parameter-specification-id', 'Mean grasping time steps', 'Mean std of mean grasping time steps'])
        for key in data_dict_mean.keys():
            w.writerow([clean_parameter_specification_id_string(key), data_dict_mean[key], data_dict_std[key]])
        pass

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
    """
        Performs 100 test runs for a given trained model. See method evaluate_measurements_per_param_specification()
        for thorough explanation.
    :param model_path: Path to a trained model.
    :param params_path: Path to the file summarizing the parameters used for training the model.
    :return:
    """

    # Run simulation 100 times for a single model

    num_test_runs = 100   # FIXME
    iterations = 1000   # FIXME

    print('Running 100 tests on model: ' + model_path)
    print('Using params: ' + params_path)

    # Load params
    params = load_model_params(params_path)

    #params['render'] = True

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
    test_scores = []        # Over 100 eval runs
    test_times_means = []   # Over 100 eval runs
    test_times_stds = []    # Over 100 eval runs

    for _ in range(num_test_runs):
        score_over_test_run = 0
        time_steps_at_previous_event = time_step_counter = 0
        reaching_times_over_test_run = []
        obs = env.reset()

        while time_step_counter < iterations:

            env.envs[0].set_step_counter(time_step_counter)
            action, _ = model.predict(obs)
            obs, _, _, info = env.step(action)  # Assumption: eval conducted on single env only!

            reward, time_step_counter, done = info[0][:]

            #print(info)
            #time.sleep(0.1)


            if done:
                #print('Accumulated rew.:' + str(score_over_test_run))
                #print('Time: ' + str(time_steps))

                if reward > 0:
                    # Gripper has reached goal position & orientation
                    score_over_test_run += reward  # Reward clipped to binary 0 | 1
                    time_steps_reaching = time_step_counter - time_steps_at_previous_event
                    reaching_times_over_test_run.append(time_steps_reaching)
                    #print('Time to goal: ' + str(time_steps_reaching))
                time_steps_at_previous_event = time_step_counter

                #print(score_over_test_run)
                #print(reaching_times_over_test_run)

                obs = env.reset()
        test_scores.append(score_over_test_run)  # Test score obtained per test run
        test_times_means.append(np.nanmean(np.array(reaching_times_over_test_run)))  # Average reaching time per test run
        test_times_stds.append(np.nanstd(np.array(reaching_times_over_test_run)))  # Std's of reaching time per test run

    # Return mean test score for model
    print('Scores:')
    print(test_scores)
    print('Times:')
    print(test_times_means)

    # Return:
    # mean test score over number of test runs for a single model,
    # mean over 100*[mean time per test run],
    # mean over 100*[std of mean time per test run]
    return np.nanmean(np.array(test_scores)), np.nanmean(np.array(test_times_means)), np.nanmean(np.array(test_times_stds))


def evaluate_measurements_per_param_specification(path, params):
    """
        Iterates through all parameter settings used during training of models and all the models trained per parameter
        setting. Parameter settings are referred to by their parameter-setting/specification-id (=ID).
        For each ID, the function iterates through all models that were trained using the ID. For each of these models,
        the function obtains the average number of successful grasps per 1000 time steps (= 1 evaluation run) computed
        over 100 evaluation games. Finally, given the mean per model computed previously, the mean per model is
        averaged over for all models belonging to a single ID (= parameter setting) in order to compute the mean number
        of grasps per 1000 time steps per ID. This value per ID is returned in dictionary eval_scores_per_param_setting.

        Also, the mean time to get into grasping position (including desired orientation of the end-effector) per ID is
        obtained by averaging over the mean times each model belonging to a given ID needs for performing a successful
        grasp. The mean time per model, again computed over 100 evaluation games, each taking 1000 time steps, is
        computed as follows:

            For each successful grasp (= successfully attaining valid graping position), the number of time steps needed
            is computed by subtracting the time step at which the simulation was reset the last time from the time step
            at which the successful grasping position is attained. For each of the 100 evaluation runs per model, the
            mean number of time steps needed for attaining the grasping position is computed by averaging all the
            numbers of time steps needed for attaining grasping position recorded for a given test run.
            In the next step, the mean number of time steps needed for attaining grasping position for all evaluation
            runs per model is averaged over in order to compute the mean number of time steps needed for attaining
            grasping position per single model.
            Again, by averaging over the mean number of time steps needed for attaining grasping position per model for
            all models belonging to a certain ID, the corresponding number of time steps per ID (= parameter setting) is
            computed.

        This value per ID is returned via dictionary mean_time_per_param_setting.

        Also the average standard deviation corresponding to the mean grasping times per ID is computed.
        This happens as follows:

            For each successful grasp, the number of time steps needed to get into grasping position is computed.
            For each of the 100 evaluation runs per model, the respective standard deviation corresponding to the mean
            number of time steps needed for attaining grasping position per test run per model is computed.
            In the next step, the standard deviation computed in the previous step [per test run per model] is averaged
            over in order to get the average standard deviation in terms of time steps per model. Then, the average
            standard deviations per model are averaged over for all models belonging to a given ID (=parameter setting).

        This value per ID is returned in std_time_per_param_setting.

    :param path: Path to folder containing the folders in which trained models are located.
    :param params: Dictionary: key = parameter-specification-id (=file-name of parameter-setting-file used for training)
                               val = list of models' folders trained using parameter-id specified as corresponding key

    :return: See above.
    """
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
    print('Param-specification-Avg-times:')
    print(mean_time_per_param_setting)
    print('Param-specification-Avg-time-avg-std:')
    print(std_time_per_param_setting)
    print()

    return eval_scores_per_param_setting, mean_time_per_param_setting, std_time_per_param_setting


candidate_dirs, list_outtakes_failure = get_complete_trials(PATH_READ)
# list_outtakes_failure == incomplete data sets

used_dict, filtered_out = remove_redundant_runs(PATH_READ, candidate_dirs)
# filtered_out == directories rejected due to max number of directories to include exceeded

# used_dict == key : parameter-specification-id ; value : list of directories (including data of a single test run)
#                                                         associated with a given parameter specification given by key

not_used_lists = [list_outtakes_failure, filtered_out]

eval_scores_per_param_setting, mean_time_per_param_setting, std_time_per_param_setting = \
    evaluate_measurements_per_param_specification(PATH_READ, used_dict)

# Save to file which data was included in final analysis and which wasn't
save_which_data_was_used(PATH_WRITE+"EvaluatedData", used_dict, not_used_lists)

save_dict_to_file(direct=PATH_WRITE+'Statistics', name='param_average_scores', data_dict=eval_scores_per_param_setting)
save_dict_to_file(direct=PATH_WRITE+'Statistics', name='param_average_time', data_dict=mean_time_per_param_setting)
save_dict_to_file(direct=PATH_WRITE+'Statistics', name='param_average_std_time', data_dict=std_time_per_param_setting)
save_mean_and_std_time_to_file(direct=PATH_WRITE+'Statistics',
                               name='param_average_time_and_avg_std',
                               data_dict_mean=mean_time_per_param_setting,
                               data_dict_std=std_time_per_param_setting)


print('Complete and sufficient runs:')
print(candidate_dirs)
print('Outtakes:')
print(not_used_lists)
print()
print('Used parameter id\'s and the test runs that used them:')
print(used_dict)


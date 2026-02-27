"""
File: reporting.py
Author: Matthew Allen

Description:
    Misc. functions to log a dictionary of metrics to logger and print them to the console.
"""


import torch
import numpy as np
import locale
locale.setlocale(locale.LC_ALL, '')


def _form_printable_groups(report):
    """
    Function to create a list of dictionaries containing the data to print to the console in a specific order.
    :param report: A dictionary containing all of the values to organize.
    :return: A list of dictionaries containing keys organized in the desired fashion.
    """

    groups = [
        {"Policy Reward": report["Policy Reward"],
         "Policy Entropy": report["Policy Entropy"],
         "Value Function Loss": report["Value Function Loss"]},

        {"Mean KL Divergence": report["Mean KL Divergence"],
         "SB3 Clip Fraction": report["SB3 Clip Fraction"],
         "Policy Update Magnitude": report["Policy Update Magnitude"],
         "Value Function Update Magnitude": report["Value Function Update Magnitude"]},

        {"Collected Steps per Second": report["Collected Steps per Second"],
         "Overall Steps per Second": report["Overall Steps per Second"]},

        {"Timestep Collection Time": report["Timestep Collection Time"],
         "Timestep Consumption Time": report["Timestep Consumption Time"],
         "PPO Batch Consumption Time": report["PPO Batch Consumption Time"],
         "Total Iteration Time": report["Total Iteration Time"]},

        {"Cumulative Model Updates": report["Cumulative Model Updates"],
         "Cumulative Timesteps": report["Cumulative Timesteps"]},

        {"Timesteps Collected": report["Timesteps Collected"]},
              ]

    return groups

def report_metrics(loggable_metrics, debug_metrics, logger=None):
    if logger is not None:
        logger.log(loggable_metrics)

def dump_dict_to_debug_string(dictionary):
    """
    Function to format the data in a loggable dictionary so the line length is limited.

    :param dictionary: Data to format.
    :return: A string containing the formatted elements of that dictionary.
    """

    debug_string = ""
    for key, val in dictionary.items():
        if type(val) == torch.Tensor:
            if len(val.shape) == 0:
                val = val.detach().cpu().item()
            else:
                val = val.detach().cpu().tolist()

        # Format lists of numbers as [num_1, num_2, num_3] where num_n is clipped at 5 decimal places.
        if type(val) in (tuple, list, np.ndarray, np.array):
            arr_str = []
            for arg in val:
                arr_str.append(locale.format_string("%7.5f", arg, grouping=True) if type(arg) == float
                               else "{},".format(arg))

            arr_str = ' '.join(arr_str)
            debug_string = "{}{}: [{}]\n".format(debug_string, key, arr_str[:-1])

        # Format floats such that only 5 decimal places are shown.
        elif type(val) in (float, np.float32, np.float64):

            debug_string = "{}{}: {}\n".format(debug_string, key, locale.format_string("%7.5f", val, grouping=True))

        # Print ints with comma separated thousands (locale aware).
        elif type(val) in (int, np.int32, np.int64):
            debug_string = "{}{}: {}\n".format(debug_string, key, locale.format_string("%d", val, grouping=True))
        # Default to just printing the value if it isn't a type we know how to format.
        else:
            debug_string = "{}{}: {}\n".format(debug_string, key, val)

    return debug_string

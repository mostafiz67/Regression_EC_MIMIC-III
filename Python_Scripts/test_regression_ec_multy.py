from itertools import combinations
from typing import List
from warnings import filterwarnings
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame
from typing_extensions import Literal
import pickle

ECMethod = Literal["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", 
                    "intersection_union_sample", "intersection_union_all", "intersection_union_distance"]
EC_METHODS: List[ECMethod] = ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed",
                    "intersection_union_sample", "intersection_union_all", "intersection_union_distance"]


def regression_ec(residuals: List[ndarray], method: ECMethod) -> List[ndarray]:
    filterwarnings("ignore", "invalid value encountered in true_divide", category=RuntimeWarning)
    consistencies = []
    for pair in combinations(residuals, 2):
        r1, r2 = pair
        r = np.vstack(pair)
        sign = np.sign(np.array(r1) * np.array(r2))
        if method == "ratio-signed":
            consistency = np.multiply(sign, np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0))
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio":
            consistency = np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0)
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio-diff-signed":
            consistency = np.multiply(sign, (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2)))
            consistency[np.isnan(consistency)] = 0
        elif method == "ratio-diff":
            consistency = (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2))
            consistency[np.isnan(consistency)] = 0
        elif method =="intersection_union_sample":
            conditions = [(r1>=0)&(r2>=0), (r1<=0)&(r2<=0)]
            choice_numerator = [np.minimum(r1, r2), np.zeros(len(r1))]
            choice_denominator = [np.maximum(r1, r2), -np.add(r1,r2)]
            numerator = np.select(conditions, choice_numerator)
            denominator = np.select(conditions, choice_denominator, np.add(np.abs(r1), np.abs(r2)))
            consistency = np.divide(numerator, denominator)
            consistency[np.isnan(consistency)] = 1
        elif method =="intersection_union_all":
            conditions = [(r1>=0)&(r2>=0), (r1<=0)&(r2<=0)]
            choice_numerator = [np.minimum(r1, r2), np.zeros(len(r1))]
            choice_denominator = [np.maximum(r1, r2), -np.add(r1,r2)]
            numerator = np.select(conditions, choice_numerator)
            denominator = np.select(conditions, choice_denominator, np.add(np.abs(r1), np.abs(r2)))
            consistency = np.divide(np.sum(numerator), np.sum(denominator)) # all sum and then divide
            consistency = np.nan_to_num(consistency, copy=True, nan=1.0)
        elif method =="intersection_union_distance":
            conditions = [(r1>=0)&(r2>=0), (r1<=0)&(r2<=0)]
            choiceValue = [np.abs(np.subtract(np.abs(r1), np.abs(r2))), np.add(np.abs(r1), np.abs(r2))]
            consistency = np.select(conditions, choiceValue, np.add(np.abs(r1), np.abs(r2)))
        else:
            raise ValueError("Invalid method")
        consistencies.append(consistency)
    filterwarnings("default", "invalid value encountered in true_divide", category=RuntimeWarning)
    return consistencies


# def calculate_ECs() -> DataFrame:
#     NB_TARGETS = 25
#     # rep_fold_residuals = np.arange(150750).reshape(10, 5025, NB_TARGETS) # shape(w, x, y) w=rep*fold; x=window; y=timepoints
#     # # transpose because want to take target wise one dimesional array
#     # transoped_residuals = np.transpose(rep_fold_residuals, (2, 0, 1)) 
#     # print(np.shape(transoped_residuals))
#     # Split the reshaped residual list into number of target array
#     PICKLE_FILENAME = "/home/mostafiz/Downloads/MIMIC-get/rep_residuals/" + "all_rep_fold_residuals_" + ("p085639") 
#     with open(PICKLE_FILENAME, 'rb') as handle:
#         transoped_residuals=pickle.load(handle)
#     print(np.shape(transoped_residuals))
#     target_residuals = np.array_split(transoped_residuals, NB_TARGETS) 
#     print(np.shape(target_residuals))
#     summaries = []
#     for method in ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed",
#                     "intersection_union_sample", "intersection_union_distance"]:
#         all_target_consistencies = []
#         for target_nb in range(0, NB_TARGETS):
#             consistencies = regression_ec(np.squeeze(target_residuals[target_nb], axis=0), method)
#             all_target_consistencies.append(consistencies)

#         # Overall mean and std across all values and method
#         df1 = pd.DataFrame({"Method": method,
#                             "EC_Mean": np.array(all_target_consistencies).mean(),
#                             "EC_Mean_sd": np.array(all_target_consistencies).mean(axis=0).std(ddof=1)}, index=[0])
#         # Mean collapsing the last two axes
#         df2 = pd.DataFrame([np.array(all_target_consistencies).mean(axis=(-2, -1))], 
#                     columns=[f'Mean_target_{i+1}' for i in range(np.array(all_target_consistencies).shape[0])])
#         # Sd of the mean across the last axis. 
#         df3 = pd.DataFrame([np.array(all_target_consistencies).mean(axis=-2).std(ddof=1, axis=-1)],
#                     columns=[f'Mean_target_{i+1}_sd' for i in range(np.array(all_target_consistencies).shape[0])])
#         final_result = pd.concat([df1, df2, df3], axis=1)
#         summaries.append(final_result)
#     summary = pd.concat(summaries, axis=0, ignore_index=True)
#     return summary

def calculate_EC_2(residuals: ndarray) -> DataFrame:
    cols = [f'Mean_target_{i+1}' for i in range(residuals.shape[0])]
    cols_sd = [f'Mean_target_{i+1}_sd' for i in range(residuals.shape[0])]
    summaries = []
    for method in ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed",
                    "intersection_union_sample", "intersection_union_distance"]:
        cs = np.stack([regression_ec(residuals[idx], method) for idx in range(residuals.shape[0])])
        # cs.shape == (N_TARGETS, K, N_WINDOWS) where K = N * (N - 1) // 2 for N = N_RESIDUALS
        print(np.shape(cs))
        df1 = DataFrame({"Method": method, "EC_Mean": cs.mean(), "EC_Mean_sd": cs.mean(axis=0).std(ddof=1)}, index=[0])
        df2 = DataFrame([cs.mean(axis=(1, 2))], columns=cols)
        df3 = DataFrame([cs.mean(axis=-2).std(ddof=1, axis=-1)], columns=cols_sd)
        summaries.append(pd.concat([df1, df2, df3], axis=1))
    summary = pd.concat(summaries, axis=0, ignore_index=True)
    return summary

if __name__=="__main__":
    # df = calculate_ECs()
    PICKLE_FILENAME = "/home/mostafiz/Downloads/MIMIC-get/rep_residuals/" + "all_rep_fold_residuals_" + ("p064965") 
    with open(PICKLE_FILENAME, 'rb') as handle:
        transoped_residuals=pickle.load(handle)
    print(np.shape(transoped_residuals))  
    df = calculate_EC_2(transoped_residuals)
    # filename = f"multy_target_error.csv"
    print(df)
    outfile = "/home/mostafiz/Downloads/MIMIC-get/rep_residuals/" + "all_rep_fold_residuals_" + ("p064965") + "d_.csv"
    df.to_csv(outfile)
    # print(f"Saved results for {args.dataset} error to {outfile}")
"""
Some code is inherited from https://stackoverflow.com/questions/71430032/how-to-compare-two-numpy-arrays-with-multiple-condition
"""

from itertools import combinations
from typing import List
from warnings import filterwarnings
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame
from typing_extensions import Literal
import pickle
from pathlib import Path

# directory-related
ROOT = Path(__file__).resolve().parent
ALL_RESIDUALS = ROOT / "all_residuals"
EC_PLOTS = ROOT / "ec_plots"

ECMethod = Literal["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", 
                    "intersection_union_sample", "intersection_union_distance"]
EC_METHODS: List[ECMethod] = ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed",
                    "intersection_union_sample", "intersection_union_distance"]
TEST_SIDS = [
        "p006365", "p013593", "p017822", "p023339", 
        "p031284","p046429", "p058242", "p064965", "p074438", 
        "p085639", "p091881", "p093117", "p098226",
    ]

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
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choice_numerator = [np.minimum(r1, r2), np.minimum(np.abs(r1), np.abs(r2))]
            choice_denominator = [np.maximum(r1, r2), np.maximum(np.abs(r1), np.abs(r2))]
            numerator = np.select(conditions, choice_numerator, np.zeros(len(r1)))
            denominator = np.select(conditions, choice_denominator, np.abs(np.add(np.abs(r1), np.abs(r2))))
            consistency = np.divide(numerator, denominator)
            consistency[np.isnan(consistency)] = 1
        elif method =="intersection_union_all":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choice_numerator = [np.minimum(r1, r2), np.minimum(np.abs(r1), np.abs(r2))]
            choice_denominator = [np.maximum(r1, r2), np.maximum(np.abs(r1), np.abs(r2))]
            numerator = np.select(conditions, choice_numerator, np.zeros(len(r1)))
            denominator = np.select(conditions, choice_denominator, np.abs(np.add(np.abs(r1), np.abs(r2))))
            consistency = np.divide(np.sum(numerator), np.sum(denominator)) # all sum and then divide
            consistency = np.nan_to_num(consistency, copy=True, nan=1.0)
        elif method =="intersection_union_distance":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choiceValue = [np.abs(np.subtract(np.abs(r1), np.abs(r2))), np.abs(np.subtract(np.abs(r1), np.abs(r2)))]
            consistency = np.select(conditions, choiceValue, np.add(np.abs(r1), np.abs(r2)))
        else:
            raise ValueError("Invalid method")
        consistencies.append(consistency)
    filterwarnings("default", "invalid value encountered in true_divide", category=RuntimeWarning)
    return consistencies

def calculate_ECs(residuals: ndarray) -> DataFrame:
    cols = [f'Mean_target_{i+1}' for i in range(residuals.shape[0])]
    cols_sd = [f'Mean_target_{i+1}_sd' for i in range(residuals.shape[0])]
    summaries = []
    for method in EC_METHODS:
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
    for sid in TEST_SIDS:
        pickle_filename = "all_rep_fold_residuals_" + (sid) 
        PICKLE_FILENAME = ALL_RESIDUALS / pickle_filename
        with open(PICKLE_FILENAME, 'rb') as handle:
            transoped_residuals=pickle.load(handle)
        print(np.shape(transoped_residuals))  
        df = calculate_ECs(transoped_residuals)
        filename = f"{sid}_error.csv"
        # print(df)
        outfile = ALL_RESIDUALS / filename
        df.to_csv(outfile)
    # print(f"Saved results for {args.dataset} error to {outfile}")
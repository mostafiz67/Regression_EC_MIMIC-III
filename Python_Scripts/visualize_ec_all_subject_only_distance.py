import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# directory-related
ROOT = Path(__file__).resolve().parent
ALL_RESIDUALS = ROOT / "all_residuals"
EC_PLOTS = ROOT / "ec_plots"

TEST_SIDS = [
        "p006365", "p013593", "p017822", "p023339", 
        "p031284","p046429", "p058242", "p064965", "p074438", 
        "p085639", "p091881", "p093117", "p098226",
    ]

for sid in TEST_SIDS:
    ecs_filename = f"{sid}_error.csv"
    outfile = ALL_RESIDUALS / ecs_filename

    plt.rcParams['figure.figsize'] = (20, 15)
    plt.style.use('fivethirtyeight')

    data = pd.read_csv(outfile, delimiter=',', header=0) # no skip needed anymore
    data = data.rename(columns=lambda x: re.sub('Mean_target','Time_point',x))
    data = data.rename(columns={"Time_point_1": "0.0hr",
                        "Time_point_2": "0.5hr",
                        "Time_point_3": "1.0hr",
                        "Time_point_4": "1.5hr",
                        "Time_point_5": "2.0hr",
                        "Time_point_6": "2.5hr",
                        "Time_point_7": "3.0hr",
                        "Time_point_8": "3.5hr",
                        "Time_point_9": "4.0hr",
                        "Time_point_10": "4.5hr",
                        "Time_point_11": "5.0hr",
                        "Time_point_12": "5.5hr",
                        "Time_point_13": "6.0hr",
                        "Time_point_14": "6.5hr",
                        "Time_point_15": "7.0hr",
                        "Time_point_16": "7.5hr",
                        "Time_point_17": "8.0hr",
                        "Time_point_18": "8.5hr",
                        "Time_point_19": "9.0hr",
                        "Time_point_20": "9.5hr",
                        "Time_point_21": "10.0hr",
                        "Time_point_22": "10.5hr",
                        "Time_point_23": "11.0hr",
                        "Time_point_24": "11.5hr",
                        "Time_point_25": "12.0hr",
                        }
                        )
    data = data.loc[data['Method'] == 'intersection_union_distance']
    data = data[["0.0hr", "2.5hr", "5.0hr", "7.5hr", "10.0hr",  "EC_Mean", "Unnamed: 0", "Method"]]
    data = data.drop(["EC_Mean", "Unnamed: 0"], axis=1).drop(data.filter(regex="sd").columns, axis=1).set_index("Method").T
    print(data)
    ax = data.plot(linewidth=2, fontsize=30, marker='o')

    # Additional customizations
    ax.set_xlabel('Time in every 30 minutes interval', fontsize=25)
    ax.set_ylabel("EC values", fontsize=25)
    plt.suptitle(f'Error Consistency (Intersection-Union-Distance) of subject {sid}', fontsize=30)
    # plt.yticks(np.arange(0, 1.05, 0.15))
    # plt.xticks(np.arange(0, 25, 5))
    # plt.xticks(rotation=70)
    plt.yticks(rotation=70)
    ax.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    filename = sid + "_distance_ec_plot_pre_tar_corr_pro_simplified.png"
    plt.savefig(EC_PLOTS / filename)
    # plt.show()

# https://stackoverflow.com/questions/58899038/how-to-open-multiple-json-file-from-folder-and-merge-them-in-single-json-file-in

import pandas as pd
import glob

ALL_SUMMARY_PATH = "/home/mostafiz/Downloads/MIMIC-get/logs/dl_logs/LSTM+prev_lact/lightning_logs/lightning_logs/version_*/summary.json"
combined = pd.DataFrame()
for json_file in glob.glob(ALL_SUMMARY_PATH): #Assuming that your json files and .py file in the same directory
    with open(json_file, "rb") as infile:
        summary_file = pd.read_json(infile)
        combined = pd.concat([combined, summary_file], axis=0)
# print(combined.describe())
# print(combined)

test = combined.filter(regex="M").describe().to_markdown(tablefmt="simple", index=True, floatfmt="0.3f")
print(
    combined.filter(regex="M").describe().to_markdown(tablefmt="simple", index=True, floatfmt="0.3f")
)
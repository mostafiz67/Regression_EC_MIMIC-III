import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# directory-related
ROOT = Path(__file__).resolve().parent
EC_PLOTS = ROOT / "ec_plots"

plt.rcParams['figure.figsize'] = (15, 10)
plt.style.use('fivethirtyeight')

outfile = "/home/mostafiz/Downloads/MIMIC-get/rep_residuals/" + "all_rep_fold_residuals_" + ("p064965")+"d_.csv"
data = pd.read_csv(outfile, delimiter=',', header=0) # no skip needed anymore
data = data.rename(columns=lambda x: re.sub('Mean_target','Time',x))
data = data.drop(["EC_Mean", "Unnamed: 0"], axis=1).drop(data.filter(regex="sd").columns, axis=1).set_index("Method").T

print(data)

ax = data.plot(linewidth=2, fontsize=12)

# Additional customizations
ax.set_xlabel('Time ')
ax.set_ylabel("EC values")
plt.yticks(np.arange(0, 1, 0.20))
ax.legend(fontsize=12)
plt.savefig("bla_p064965.png")
plt.show()

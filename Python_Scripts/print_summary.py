from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--corrs", action="store_true")
    args = parser.parse_args()
    parent = Path(args.input)
    while "version" not in parent.stem:
        parent = parent.parent
    print(f"Looking in {parent} for 'summary.json'")
    path = sorted(parent.rglob("summary.json"))[0]
    df = pd.read_json(path)
    print(df)
    test=pd.DataFrame()
    test = df.filter(regex="M").describe().to_markdown(tablefmt="simple", index=True, floatfmt="0.3f")
    print(
        df.filter(regex="M").describe().to_markdown(tablefmt="simple", index=True, floatfmt="0.3f")
    )
    if args.corrs:
        print(
            df.filter(regex="r")
            .describe()
            .to_markdown(tablefmt="simple", index=True, floatfmt="0.3f")
        )

# TMR, Coefficient of Variation

import pandas as pd
import glob
import os

graph_dir = "experiments/zexp_5.1_plots/results"

# 1) Map each “provider” key to its experiment folder
exp_dirs = {
    "google":      "experiments/exp_20250618_164751_262ed75f",
    "openai":      "experiments/exp_20250618_164830_e51d23bd",
    "anthropic":   "experiments/exp_20250619_192224_1a594ae5",
    "azure":       "experiments/exp_20250618_164345_f0bbdb78",
    # "azure":       "experiments/exp_20250623_172221_2ec88f3f",
    "aws":         "experiments/exp_20250618_164129_04a19ed7",
    # "aws":         "experiments/exp_20250623_172333_96426214",
    # "aws_2":         "experiments/exp_20250620_014003_b1b5a923"
    "togetherai":  "experiments/exp_20250618_184735_750e2005",
    # "togetherai":  "experiments/exp_20250623_172417_3fece75c"
    # "cloudflare":  "experiments/exp_20250618_165521_fcdfe7ed",
    "cloudflare": "experiments/exp_20250625_231541_f3c90acf",
    "vllm":        "experiments/exp_20250618_165035_d0f540aa",
}

# 2) Read each folder’s CSVs into a dict of lists-of-DataFrames
dfs = {}
all_dfs = {}
skip_idxs = {0, 3, 7}
for name, folder in exp_dirs.items():
    pattern = os.path.join(folder, "*.csv")
    paths   = glob.glob(pattern)
    all_dfs[name] = [pd.read_csv(p) for p in paths]
    dfs[name] = [
        df
        for idx, df in enumerate(all_dfs[name])
        if idx not in skip_idxs
    ]
    for i, df in enumerate(dfs[name]):
        print(i, df['metric'].unique(), df.shape)
    print(f"{name}: loaded {len(dfs[name])} CSV(s) from {folder}")

# 3) Build dfs_all by zipping in the same provider-order
provider_order = list(exp_dirs.keys())  # e.g. ["google", "openai", …, "vllm"]
# get list-of-lists in order:
list_of_df_lists = [dfs[p] for p in provider_order]

# 4) Concatenate the i-th file of each provider across all providers
dfs_all = [
    pd.concat(dfs_tuple, ignore_index=True)
    for dfs_tuple in zip(*list_of_df_lists)
]

print(f"Built {len(dfs_all)} combined DataFrames in dfs_all")

# Remove total_tokens and accuracy as before
total_tokens = dfs_all.pop(4)
dpsk_output = dfs_all.pop(6)
for i, df in enumerate(dfs_all):
    print(df.head(1))
# want to remove entries with model = "common-model-small"
    print(df['metric'].unique())
    print(df['provider'].unique())
    print(df['model'].unique())
    print(df['max_output'].unique())
    print(f"combined #{i}: {df.shape}")
    print("-----------------")

# 1) Build a dict of your metric‐DataFrames by name
metric_dfs = { df['metric'].iloc[0] : df for df in dfs_all }

# 2) Pull out the median & p99 frames
df_med  = metric_dfs['timebetweentokens_median'].reset_index(drop=True)
df_p95  = metric_dfs['timebetweentokens_p95'].reset_index(drop=True)

# 3) Group and compute mean values
med_means = df_med.groupby('provider')['value'].mean().rename('median_mean_tbt')
p95_means = df_p95.groupby('provider')['value'].mean().rename('p95_mean_tbt')

# 4) Combine and compute ratio
summary = pd.concat([p95_means, med_means], axis=1).reset_index()
summary['tmr'] = summary['p95_mean_tbt'] / summary['median_mean_tbt']

# 5) Output
print("TMR Table")
print(summary.to_markdown(index=False))

# Optional: save
os.makedirs(graph_dir, exist_ok=True)
summary.to_csv(os.path.join(graph_dir, "tmr_tbt.csv"), index=False)

df = metric_dfs['timetofirsttoken'].reset_index(drop=True)

stats = (
    df[df['metric'].isin(["timetofirsttoken","response_times"])]
    .groupby(['provider','input_size'])['value']
    .quantile([0.5, 0.95, 0.99])
    .unstack(level=-1)
    .rename(columns={0.5:'median',0.95:'p95',0.99:'p99'})
)

stats['tmr'] = stats['p95'] / stats['median']

print(stats)
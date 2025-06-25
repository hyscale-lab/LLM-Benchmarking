import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.lines import Line2D
from datetime import datetime
import re
import ast

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
    "cloudflare":  "experiments/exp_20250618_165521_fcdfe7ed",
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

# 3) Make sure they align on the same “keys”:
#    if each row is a distinct request you can join on index…
ratio_df = pd.DataFrame({
    'provider': df_med['provider'],
    'model':    df_med['model'],
    'input_size': df_med['input_size'],
    'median':   df_med['value'],
    'p95':      df_p95['value']
})

# 4) Compute the ratio (p99 over median)
ratio_df['p95_to_median'] = ratio_df['p95'] / ratio_df['median']

# 5) (Optional) Inspect it
print(ratio_df.head())
print(ratio_df.groupby('provider')['p95'].describe()['mean'])
print(ratio_df.groupby('provider')['median'].describe()['mean'])
print(ratio_df.groupby('provider')['p95_to_median'].describe()['mean'])

ratio_summary = (
    ratio_df
    .groupby('provider', as_index=False)['p95_to_median']
    .mean()
    .rename(columns={'p95_to_median':'mean_p95_to_median'})
)
print(ratio_summary)

# out_path = os.path.join(graph_dir, "tmr_df.csv")
# ratio_df.groupby('provider')['p99_to_median'].describe().to_csv(out_path, index=False)
# print(f"Saved full ratio table to {out_path}")
# i want to for each request a ratio between maybe as another df - df['timebetweentokens_median'] and df['timebetweentokens_p99'] 

# accuracy = dfs_all.pop(3)
# 1) Group by provider and compute the two means
summary = (
    ratio_df
    .groupby('provider', as_index=False)
    .agg(
        p95_mean_tbt    = ('p95',    'mean'),
        median_mean_tbt = ('median', 'mean')
    )
)

# 2) Compute the “p99:median ratio” of those means
summary['tmr'] = summary['p95_mean_tbt'] / summary['median_mean_tbt']

# 3) (Optional) Pretty–print as Markdown
print(summary.to_markdown(index=False))
# out_path = os.path.join(graph_dir, "tmr_df.csv")
# summary.to_csv(out_path, index=False)
# print(f"Saved full ratio table to {out_path}")

# Filter to only include the desired metrics
desired_metrics = ["timetofirsttoken","timebetweentokens", "response_times"]
# desired_metrics = ["timebetweentokens"]
dfs_filtered = []

for df in dfs_all:
    metric = df['metric'].iloc[0]
    if metric in desired_metrics:
        dfs_filtered.append(df)

# Use filtered dataframes
dfs_all = dfs_filtered
colors = {
    'vLLM': 'black',
    'AWSBedrock': 'teal',
    'Azure': '#FF8000',
    'Cloudflare': 'green',
    'Open_AI': '#D22F2D',
    'TogetherAI': 'purple',
    'Anthropic': '#e377c2',
    'GoogleGemini': '#007FFF'
}

# Create figure with horizontal subplots (now only 3 subplots)
fig, axes = plt.subplots(1, len(dfs_all), figsize=(13, 3.5), sharey=True)

# Set global font sizes
plt.rcParams.update({
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 18
})

display_titles = {
    'timebetweentokens': 'TBT',
    'timetofirsttoken': 'TTFT',
    'response_times': 'TRT'
}

# Plot each metric in its own subplot
for i, (ax, df) in enumerate(zip(axes, dfs_all)):
    print(f"Processing subplot {i}: {df['metric'].iloc[0]}")
    
    for provider in df['provider'].unique():
    # for model in df['model'].unique():
        subset = df[df['provider'] == provider]
        # subset = df[df['model'] == model]
        if subset.empty:
            continue

        values = subset['value'].values

        if df['metric'].iloc[0] == "timebetweentokens":
            flattened_values = []
            print(len(values[0]))
            for sublist in values:
                if isinstance(sublist, str):
                    try:
                        # Safely evaluate string as Python literal
                        parsed_list = ast.literal_eval(sublist)
                        flattened_values.extend([float(x) for x in parsed_list])
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing: {sublist[:50]}... Error: {e}")
                        continue
                else:
                    flattened_values.extend([float(x) for x in sublist])
            
            values = np.array(flattened_values) * 1000
        else:
            values = subset['value'].values * 1000 

        if df['metric'].iloc[0] == "timetofirsttoken" or df['metric'].iloc[0] == "response_times":
            # print("STATS: ")
            # print(df['metric'].iloc[0])
            # print(provider)
            # vals_ = subset['value'].values
            # # print(vals_[0])
            # vals = np.sort(vals_)
            # # print(vals[-1])
            # median = np.percentile(vals, 50)
            # p95 = np.percentile(vals, 95)
            # p99 = np.percentile(vals, 99)
            # print("Median: ", median, "p95: ", p95, "p99: ", p99)
            # print("---------------")
             
            stats = (
                df[df['metric'].isin(["timetofirsttoken","response_times"])]
                .groupby(['provider','input_size'])['value']
                .quantile([0.5, 0.95, 0.99])
                .unstack(level=-1)
                .rename(columns={0.5:'median',0.95:'p95',0.99:'p99'})
            )
            print(stats)
              
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        # Plot with different line widths for vLLM
        if provider == 'vLLM':
            ax.plot(sorted_vals, cdf, color=colors[provider], linewidth=2.4, label=provider)
        else:
            ax.plot(sorted_vals, cdf, color=colors[provider], linewidth=1.8, label=provider)
    
    # Set x-axis to log scale
    ax.set_xscale("log")
    
    # Generate clean title
    metric_raw = df['metric'].iloc[0]
    metric = re.sub(r'([a-z])([A-Z])', r'\1 \2', metric_raw.replace('_', ' ')).title()
    ax.set_title(display_titles.get(metric_raw, metric_raw), fontsize=16, pad=20)
    
    # Set x-label
    ax.set_xlabel("Latency (ms)", fontsize=16)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Remove y-axis labels for all but the first subplot
    if i > 0:
        ax.set_ylabel('')
    else:
        ax.set_ylabel("CDF", fontsize=18)

# Set y-axis limits for all subplots
axes[0].set_ylim(0, 1.01)

# Create legend outside the plot area
# Get handles and labels from the first subplot (they should be the same for all)
handles, labels = axes[0].get_legend_handles_labels()

color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=provider) 
                               for provider, color in colors.items()]
        
legend1 = fig.legend(
    handles=color_legend_elements,
    title='Provider',
    loc='center left',
    bbox_to_anchor=(0.86, 0.5),
    # loc="best",
    # x0, y0, width, height  (in axes‐fraction coords)
    # bbox_to_anchor=(1.0, 0.7, 0.5, 0.3),
    # bbox_to_ancho / .r=(0.1, 0.5),
    fontsize=13,
    title_fontsize=14,
    frameon=True,
    borderpad=0.5,     # space between text and frame
    labelspacing=0.4,  # vertical space between entries
    columnspacing=0.6, # horizontal space between columns
    handlelength=1.0,  # length of the little line in the legend
    handletextpad=0.5, # space between line and label text
    fancybox=False,
    shadow=False,
    framealpha=1.0
    # mode='expand'      # stretch entries to fill the width
)

# Adjust layout to prevent overlap

plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Save the combined plot
current_time = datetime.now().strftime("%y%m%d_%H%M")
filename = f"selected_metrics_{current_time}.pdf"
filepath = os.path.join(graph_dir, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved selected metrics graph: {filepath}")
print(f"Included metrics: {[df['metric'].iloc[0] for df in dfs_all]}")

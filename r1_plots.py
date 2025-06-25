import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogLocator, FuncFormatter
from matplotlib.lines import Line2D
from datetime import datetime
import re
import ast

graph_dir = "experiments/zexp_5.2_deepseek_plots/results/tokens"

exp_dirs = {
    "aws": "experiments/exp_20250618_162522_20f4040a",
    "azure": "experiments/exp_20250618_162137_e31b40f5",
    "vllm": "experiments/exp_20250619_112437_c70b32bc",
    "tgt": "experiments/exp_20250618_163312_f98e83f7"
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
        print(i, df['metric'].unique())
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
#     # want to remove entries with model = "common-model-small"
    print(df['metric'].unique())
    print(df['provider'].unique())
    print(df['model'].unique())
    print(df['max_output'].unique())
    print(df['input_size'].unique())
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
out_path = os.path.join(graph_dir, "tmr_df.csv")
summary.to_csv(out_path, index=False)
print(f"Saved full ratio table to {out_path}")


desired_metrics = ["timetofirsttoken", "response_times", "timebetweentokens"]
dfs_filtered = []

for df in dfs_all:
    metric = df['metric'].iloc[0]
    if metric in desired_metrics:
        dfs_filtered.append(df)

# Use filtered dataframes
dfs_all = dfs_filtered

for i, df in enumerate(dfs_all):
    print(df.head(1))
#     # want to remove entries with model = "common-model-small"
    print(df['metric'].unique())
    print(df['provider'].unique())
    print(df['model'].unique())
    print(df['max_output'].unique())
    print(df['input_size'].unique())
    print(f"combined #{i}: {df.shape}")
    print("-----------------")

fig, axes = plt.subplots(1, len(dfs_all), figsize=(13, 3.5), sharey=True)
# plt.figure(figsize=(7, 3.5))

display_titles = {
'timebetweentokens': 'TBT',
'timetofirsttoken': 'TTFT',
'response_times': 'TRT'
}

# linestyles = {
#     1000: 'solid',
#     10000: 'dashed',
#     100000: 'dashdot'
# }

colors = {
'vLLM': 'black',    
'Azure': '#FF8000',
'AWSBedrock': 'teal',
'TogetherAI': 'purple'
}   

plt.rcParams.update({
'axes.labelsize': 16,   # X/Y label font size
'xtick.labelsize': 18,  # X‐tick label font size
'ytick.labelsize': 18   # Y‐tick label font size
})

for i, (ax, df) in enumerate(zip(axes, dfs_all)):
    for provider in df['provider'].unique():
        subset = df[
            (df['provider'] == provider)
        ]
        if subset.empty:
            continue
        values = subset['value'].values # convert to ms
        # values = subset['value'].values * 1000  # convert to ms
        # sorted_vals = np.sort(values)
#             if df['metric'].iloc[0] == "timetofirsttoken" or df['metric'].iloc[0] == "response_times":
#                 print("STATS: ")
#                 print(df['metric'].iloc[0])
#                 print(provider, input_size)
#                 vals_ = subset['value'].values
#                 # print(vals_[0])
#                 vals = np.sort(vals_)
#                 # print(vals[-1])
#                 median = np.percentile(vals, 50)
#                 p95 = np.percentile(vals, 95)
#                 p99 = np.percentile(vals, 99)
#                 print("Median: ", median, "p95: ", p95, "p99: ", p99)
#                 print("---------------")

        if df['metric'].iloc[0] == "timebetweentokens":
            flattened_values = []
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
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        if df['metric'].iloc[0] == "response_times":
            plt.subplots_adjust(right=0.8)
        
        # Plot without label to avoid cluttered legend
        if provider == 'vLLM':
            ax.plot(sorted_vals, cdf, color=colors[provider], linewidth=2.2)
        else:
            ax.plot(sorted_vals, cdf, color=colors[provider], linewidth=1.5)

    # Set x-axis to log scale
    ax.set_xscale("log")

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
        ax.set_ylabel("CDF", fontsize=15)

    # Set y-axis limits for all subplots
    axes[0].set_ylim(0, 1.02)

    handles, labels = axes[0].get_legend_handles_labels()

    # Create legend below the subplots
    # fig.legend(handles, labels, 
    #           loc='best', 
    #           bbox_to_anchor=(0.5, 0.95),
    #           ncol=len(colors), 
    #           fontsize=14,
    #           frameon=True,
    #           borderpad=0.5,
    #           columnspacing=1.0,
    #           handlelength=2.0,
#           handletextpad=0.5)
color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=provider) 
                            for provider, color in colors.items()]

# linestyle_legend_elements = [Line2D([0], [0], color='black', linestyle=style, lw=2, label=f'{size:,}') 
#                         for size, style in linestyles.items()]
        
# legend1 = plt.legend(
#     handles=color_legend_elements,
#     title='Provider',
#     loc='upper right',
#     # loc="best",
#     # x0, y0, width, height  (in axes‐fraction coords)
#     # bbox_to_anchor=(1.0, 0.7, 0.5, 0.3),
#     # bbox_to_ancho / .r=(0.1, 0.5),
#     fontsize=13,
#     title_fontsize=14,
#     frameon=True,
#     borderpad=0.5,     # space between text and frame
#     labelspacing=0.4,  # vertical space between entries
#     columnspacing=0.6, # horizontal space between columns
#     handlelength=1.0,  # length of the little line in the legend
#     handletextpad=0.5, # space between line and label text
#     fancybox=False,
#     shadow=False,
#     framealpha=1.0
#     # mode='expand'      # stretch entries to fill the width
# )
# #         legend1 = plt.legend(
# #             handles=color_legend_elements,
# #             title='Provider',
# #             loc='upper left',
# #             # x0, y0, width, height  (in axes‐fraction coords)
# #             bbox_to_anchor=(1.0, 0.7, 0.5, 0.3),
# #             fontsize=14,
# #             title_fontsize=15,
# #             frameon=True,
# #             borderpad=0.5,     # space between text and frame
# #             labelspacing=0.4,  # vertical space between entries
# #             columnspacing=0.6, # horizontal space between columns
# #             handlelength=1.0,  # length of the little line in the legend
# #             handletextpad=0.5, # space between line and label text
# #             fancybox=False,
# #             shadow=False,
# #             framealpha=1.0
# #             # mode='expand'      # stretch entries to fill the width
# #         )

    
# legend2 = plt.legend(handles=linestyle_legend_elements, title='Input Size', 
#                     loc='center right', bbox_to_anchor=(1.02, 0.1), 
#                     fontsize=13, title_fontsize=14, frameon=True, borderpad=0.5,     # space between text and frame
#     labelspacing=0.4,  # vertical space between entries
#     columnspacing=0.6, # horizontal space between columns
#     handlelength=1.0,  # length of the little line in the legend
#     handletextpad=0.5,
#                     fancybox=False, shadow=False, framealpha=1.0)

legend1 = fig.legend(
    handles=color_legend_elements,
    title='Provider',
    loc='center left',
    bbox_to_anchor=(0.86, 0.5),
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

# legend2 = fig.legend(
#     handles=linestyle_legend_elements, 
#     title='Input Size', 
#     loc='center left',
#     bbox_to_anchor=(0.86, 0.3),  # Position below the first legend
#     fontsize=13, 
#     title_fontsize=14, 
#     frameon=True, 
#     borderpad=0.5,
#     labelspacing=0.4,
#     handlelength=1.0,
#     handletextpad=0.5,
#     fancybox=False, 
#     shadow=False, 
#     framealpha=1.0
# )

# Add the first legend back (matplotlib removes it when adding the second)
# plt.gca().add_artist(legend1)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.subplots_adjust(right=0.85)  # Make room for legend at top

# Save the combined plot
current_time = datetime.now().strftime("%y%m%d_%H%M")
filename = f"selected_metrics_{current_time}.pdf"
filepath = os.path.join(graph_dir, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved selected metrics graph: {filepath}")
print(f"Included metrics: {[df['metric'].iloc[0] for df in dfs_all]}")


# for i, df in enumerate(dfs_all):
#     print(df.head(1))
# #     # want to remove entries with model = "common-model-small"
#     print(df['metric'].unique())
#     print(df['provider'].unique())
#     print(df['model'].unique())
#     print(df['max_output'].unique())
#     print(f"combined #{i}: {df.shape}")


# colors = {
#     'vLLM':       'black',
#     'Azure':      '#FF8000',
#     'AWSBedrock': 'teal',
#     'TogetherAI': 'purple'
# }

# plt.rcParams.update({
#     'axes.labelsize': 18,   # X/Y label font size
#     'xtick.labelsize': 18,  # X‐tick label font size
#     'ytick.labelsize': 18   # Y‐tick label font size
#     })


# plt.figure(figsize=(7, 5))


# # Loop over each provider
# for provider in total_tokens['provider'].unique():
#     subset = total_tokens[total_tokens['provider'] == provider]
#     if subset.empty:
#         continue

#     # Sort token counts and compute CDF
#     tokens = np.sort(subset['value'].values)
#     cdf    = np.arange(1, len(tokens) + 1) / len(tokens)

#     # Plot—use thicker line for vLLM if you like
#     lw = 2.4 if provider == 'vLLM' else 1.8
#     plt.plot(tokens, cdf,
#              color=colors.get(provider, 'gray'),
#              linewidth=lw,
#              label=provider)

# # Axis labels and grid
# plt.xlabel("Total Tokens", fontsize=18)
# plt.ylabel("CDF", fontsize=18)
# plt.grid(True, which='both', ls='--', alpha=0.3)
# plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

# # Legend inside the plot (upper left corner here)
# plt.legend(title="Provider", loc="best", fontsize=14, title_fontsize=14)
# plt.tight_layout()
# current_time = datetime.now().strftime("%y%m%d_%H%M")
# filename = f"{total_tokens['metric'].unique()[0]}_{current_time}.pdf"
# filepath = os.path.join(graph_dir, filename)
# plt.savefig(filepath)
# plt.close()

# print(f"Saved graph: {filepath}")

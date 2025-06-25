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

exp_dirs = {
    "aws": "experiments/exp_20250620_110503_06a15fc4",
    "aws_5k": "experiments/aws_1k",
    "aws_10k": "experiments/aws_5k",
    "vllm_azure": "experiments/exp_20250620_013929_b5497e18",
    "azure_5k": "experiments/azure_5k_new",
    "azure_10k": "experiments/azure_10k_new",
    "togetherai": "experiments/exp_20250620_052224_67850b68",
    "togetherai_5k": "experiments/togetherai_5k",
    "togetherai_10k": "experiments/togetherai_10k"
}

graph_dir = "experiments/zexp_5.2_output"

# 2) Read each folder’s CSVs into a dict of lists-of-DataFrames
dfs = {}
all_dfs = {}
first_provider = list(exp_dirs.keys())[0]   # e.g. "aws"
last_provider = list(exp_dirs.keys())[-1]

skip_idxs = {0, 3, 7}
for name, folder in exp_dirs.items():
    pattern = os.path.join(folder, "*.csv")
    paths   = glob.glob(pattern)
    print(name)
    for p in paths:
        try:
            print(p)
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Error in file: {p}")
            raise

    all_dfs[name] = [pd.read_csv(p) for p in paths]
    # dfs[name] = [
    #     df
    #     for idx, df in enumerate(all_dfs[name])
    #     if idx not in skip_idxs
    # ]
    dfs[name] = []
    for idx, df in enumerate(all_dfs[name]):
        if idx in skip_idxs:
            continue

        # if this is the first provider, take only the first 100 rows
        if name == first_provider:
            dfs[name].append(df.head(100))
        # if name == last_provider:
        #     dfs[name].append(df.head(201))
        elif name == "togetherai":
            dfs[name].append(df.head(200))
        elif name == "vllm_azure":
            dfs[name].append(df.head(583))
        else:
            dfs[name].append(df)
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
# for i, df in enumerate(dfs_all):
#     print(df.head(1))
# #     # want to remove entries with model = "common-model-small"
#     print(df['metric'].unique())
#     print(df['provider'].unique())
#     print(df['model'].unique())
#     print(df['max_output'].unique())
#     print(df['input_size'].unique())
#     print(f"combined #{i}: {df.shape}")
#     print("-----------------")

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
# for provider in provider
metric_dfs = { df['metric'].iloc[0] : df for df in dfs_all }

# 2) Pull out the median & p99 frames
df_med  = metric_dfs['timebetweentokens_median'].reset_index(drop=True)
df_p95  = metric_dfs['timebetweentokens_p95'].reset_index(drop=True)

# 3) Make sure they align on the same “keys”:
#    if each row is a distinct request you can join on index…
ratio_df = pd.DataFrame({
    'provider': df_med['provider'],
    'model':    df_med['model'],
    'max_output': df_med['max_output'],
    'median':   df_med['value'],
    'p95':      df_p95['value']
})

# 4) Compute the ratio (p99 over median)
ratio_df['p95_to_median'] = ratio_df['p95'] / ratio_df['median']

# 5) (Optional) Inspect it
# print(ratio_df.head())
# print(ratio_df.groupby(['provider','max_output'])['p95'].describe()['mean'])
# print(ratio_df.groupby(['provider','max_output'])['median'].describe()['mean'])
# print(ratio_df.groupby(['provider','max_output'])['p95_to_median'].describe()['mean'])

ratio_summary = (
    ratio_df
    .groupby('provider', as_index=False)['p95_to_median']
    .mean()
    .rename(columns={'p95_to_median':'mean_p95_to_median'})
)
# print(ratio_summary)

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
# print(summary.to_markdown(index=False))

# 1) Group by provider AND max_output, computing the two means
summary_by_output = (
    ratio_df
    .groupby(['provider','max_output'], as_index=False)
    .agg(
        p95_mean_tbt    = ('p95',    'mean'),
        median_mean_tbt = ('median', 'mean')
    )
)

# 2) Compute the “p99:median ratio” of those means
summary_by_output['tmr'] = (
    summary_by_output['p95_mean_tbt'] 
  / summary_by_output['median_mean_tbt']
)

# 3) Print as Markdown table
print("TMR TABLE")
print(summary_by_output.to_markdown(index=False))

# print(ratio_df[['provider','max_output','p95_to_median']].head())

out_path = os.path.join(graph_dir, "tmr_df.csv")
summary_by_output.to_csv(out_path, index=False)
print(f"Saved full ratio table to {out_path}")

desired_metrics = ['timetofirsttoken', 'response_times', "timebetweentokens"]
dfs_filtered = []

for df in dfs_all:
    metric = df['metric'].iloc[0]
    if metric in desired_metrics:
        dfs_filtered.append(df)

# Use filtered dataframes
dfs_all = dfs_filtered

# for i, df in enumerate(dfs_all):
#     print(df.head(1))
fig, axes = plt.subplots(1, len(dfs_all), figsize=(13, 3.5), sharey=True)
# plt.figure(figsize=(7, 3.5))

display_titles = {
'timebetweentokens': 'TBT',
'timetofirsttoken': 'TTFT',
'response_times': 'TRT'
}

linestyles = {
    500: 'solid',
    1000: 'dashed',
    5000: 'dashdot',
    10000: ':'
}

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

# pure_exp = total_tokens[(total_tokens['max_output'] - total_tokens['value']).abs() <= 200]

# for idx in range(len(dfs_all)):
#     print(dfs_all[idx].head(1))
#     print(dfs_all[idx].shape)
#     dfs_all[idx] = dfs_all[idx].loc[pure_exp.index]
#     print(dfs_all[idx].shape)

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

graph_dir = "experiments/zexp_5.2_output"

for i, (ax, df) in enumerate(zip(axes, dfs_all)):
    for provider in df['provider'].unique():
        for max_output in sorted(df["max_output"].unique()):
            subset = df[
                (df['provider'] == provider) &
                (df['max_output'] == max_output)
            ]
            if subset.empty:
                continue
            values = subset['value'].values # convert to ms
            # values = subset['value'].values * 1000  # convert to ms
            # sorted_vals = np.sort(values)
            if df['metric'].iloc[0] == "timetofirsttoken" or df['metric'].iloc[0] == "response_times":
                print("STATS: ")
                print(df['metric'].iloc[0])
                print(provider, max_output)
                vals_ = subset['value'].values
                # print(vals_[0])
                vals = np.sort(vals_)
                # print(vals[-1])
                median = np.percentile(vals, 50)
                p95 = np.percentile(vals, 95)
                p99 = np.percentile(vals, 99)
                print("Median: ", median, "p95: ", p95, "p99: ", p99)
                print("---------------")

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
                ax.plot(sorted_vals, cdf, color=colors[provider], linestyle=linestyles[max_output], linewidth=2.2)
            else:
                ax.plot(sorted_vals, cdf, color=colors[provider], linestyle=linestyles[max_output], linewidth=1.5)
    
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

color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=provider) 
                            for provider, color in colors.items()]

linestyle_legend_elements = [Line2D([0], [0], color='black', linestyle=style, lw=2, label=f'{size:,}') 
                        for size, style in linestyles.items()]

legend1 = fig.legend(
    handles=color_legend_elements,
    title='Provider',
    loc='center left',
    bbox_to_anchor=(0.86, 0.7),  # x, y coordinates (1.02 puts it outside)
    fontsize=13,
    title_fontsize=14,
    frameon=True,
    borderpad=0.5,
    labelspacing=0.4,
    handlelength=1.0,
    handletextpad=0.5,
    fancybox=False,
    shadow=False,
    framealpha=1.0
)

legend2 = fig.legend(
    handles=linestyle_legend_elements, 
    title='Max Output Length', 
    loc='center left',
    bbox_to_anchor=(0.86, 0.3),  # Position below the first legend
    fontsize=13, 
    title_fontsize=14, 
    frameon=True, 
    borderpad=0.5,
    labelspacing=0.4,
    handlelength=3.2,
    handletextpad=0.5,
    fancybox=False, 
    shadow=False, 
    framealpha=1.0
)

# Add the first legend back (matplotlib removes it when adding the second)
plt.gca().add_artist(legend1)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.subplots_adjust(right=0.85)  # Make room for legend at top

# Save the combined plot
current_time = datetime.now().strftime("%y%m%d_%H%M")
filename = f"selected_metrics_{current_time}.png"
filepath = os.path.join(graph_dir, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved selected metrics graph: {filepath}")
print(f"Included metrics: {[df['metric'].iloc[0] for df in dfs_all]}")

# for i, df in enumerate(dfs_all):
#     print(df.head(1))

#     plt.figure(figsize=(6.5, 3))

#     linestyles = {
#         500: 'solid',
#         1000: 'dashed',
#         5000: 'dashdot',
#         10000: ':'
#     }

#     colors = {
#     'vLLM': 'black',    
#     'Azure': '#FF8000',
#     'AWSBedrock': 'teal',
#     'TogetherAI': 'purple'
#     }   

#     plt.rcParams.update({
#     'axes.labelsize': 16,   # X/Y label font size
#     'xtick.labelsize': 18,  # X‐tick label font size
#     'ytick.labelsize': 18   # Y‐tick label font size
#     })

#     for provider in df['provider'].unique():
#         for max_output in sorted(df['max_output'].unique()):
#             subset = df[
#                 (df['provider'] == provider) &
#                 (df['max_output'] == max_output)
#             ]
#             if subset.empty:
#                 continue
#             values = subset['value'].values * 1000  # convert to ms
#             sorted_vals = np.sort(values)
#             cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            
#             # Plot without label to avoid cluttered legend
#             if provider == 'vLLM':
#                 plt.plot(sorted_vals, cdf, color=colors[provider], linestyle=linestyles[max_output], linewidth=2.2)
#             else:
#                 plt.plot(sorted_vals, cdf, color=colors[provider], linestyle=linestyles[max_output], linewidth=1.5)

#     # Create custom legends only for the first plot (for research paper)
#     if i == 0:  # Only add legends to the first plot
#         # Color legend (for providers)
#         color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=provider) 
#                                for provider, color in colors.items()]
        
#         linestyle_legend_elements = [Line2D([0], [0], color='black', linestyle=style, lw=2, label=f'{size:,}') 
#                                    for size, style in linestyles.items()]
        
#         legend1 = plt.legend(
#             handles=color_legend_elements,
#             title='Provider',
#             loc='upper left',
#             # x0, y0, width, height  (in axes‐fraction coords)
#             bbox_to_anchor=(1.0, 0.7, 0.5, 0.3),
#             fontsize=14,
#             title_fontsize=15,
#             frameon=True,
#             borderpad=0.5,     # space between text and frame
#             labelspacing=0.4,  # vertical space between entries
#             columnspacing=0.6, # horizontal space between columns
#             handlelength=1.0,  # length of the little line in the legend
#             handletextpad=0.5, # space between line and label text
#             fancybox=False,
#             shadow=False,
#             framealpha=1.0
#             # mode='expand'      # stretch entries to fill the width
#         )
  
        
#         legend2 = plt.legend(handles=linestyle_legend_elements, title='Input Size', 
#                             loc='center left', bbox_to_anchor=(1.02, 0.1), 
#                             fontsize=14, title_fontsize=15, frameon=True, borderpad=0.5,     # space between text and frame
#             labelspacing=0.4,  # vertical space between entries
#             columnspacing=0.6, # horizontal space between columns
#             handlelength=1.0,  # length of the little line in the legend
#             handletextpad=0.5,
#                             fancybox=False, shadow=False, framealpha=1.0)
        
#         # Add the first legend back (matplotlib removes it when adding the second)
#         plt.gca().add_artist(legend1)

#     plt.xlabel("Latency (ms)")
#     plt.ylabel("CDF")
    
#     # Smart title generation for compound words and underscores
#     metric_raw = df['metric'].iloc[0]
#     # Split on underscores and add spaces before capital letters in compound words
#     metric = re.sub(r'([a-z])([A-Z])', r'\1 \2', metric_raw.replace('_', ' ')).title()
#     # plt.title(f"{metric} CDF by Provider and Output Length")
#     if df['metric'].iloc[0] != "timebetweentokens":
#         plt.xscale("log")
#     plt.grid(True, alpha=0.3)
    
#     # Adjust layout to accommodate external legends only for first plot
#     plt.tight_layout()
#     if i == 0:
#         plt.subplots_adjust(right=0.68)  # Make room for legends on the right
#     plt.show()
#     current_time = datetime.now().strftime("%y%m%d_%H%M")
#     filename = f"{df['metric'].unique()[0]}_{current_time}.png"
#     filepath = os.path.join(graph_dir, filename)
#     plt.savefig(filepath)
#     plt.close()

#     print(f"Saved graph: {filepath}")

# print(pure_exp.head(1))
# print(pure_exp.shape, total_tokens.shape)
# print(pure_exp['provider'].unique())
# print(pure_exp['max_output'].unique())

# for i, df in enumerate(dfs_all):
#     print(df.head(1))
#     print(df['provider'].unique())
#     print(df['max_output'].unique())
#     print(f"combined #{i}: {df.shape}")






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

graph_dir = "experiments/zexp_5.2_modelsize_plots/results/results"

exp_dirs = {
    "azure_8b":       "experiments/exp_20250618_164345_f0bbdb78",
    "azure_70b":       "experiments/exp_20250618_171105_1492bb31",
    "azure_671b":       "experiments/exp_20250618_172700_afc8560c",
    "aws_8b":       "experiments/exp_20250618_164129_04a19ed7",
    "aws_70b":       "experiments/exp_20250618_171056_ae2f18a5",
    "aws_671b":       "experiments/exp_20250618_172744_195ee80d",
    "tgt_8b":       "experiments/exp_20250618_184735_750e2005",
    "tgt_70b":       "experiments/exp_20250618_191558_a419f85d",
    "tgt_671b":       "experiments/exp_20250619_202834_35e81537",
    "vllm_8b":       "experiments/exp_20250618_165035_d0f540aa",
    "vllm_70b":       "experiments/exp_20250618_171748_8f758800",
    "vllm_671b":       "experiments/exp_20250619_111919_5ccb399a"
}

# display_names = {
#     'common-model-small': 'LLama 3.1 8B',
#     'common-model': 'LLama 3.3 70B',
#     'deepseek-r1': 'Deepseek R1 671B',
#     'meta-llama/Llama-3.1-8B-Instruct': 'LLama 3.1 8B',
#     '/dataset/shared_models/llama3_3-70b': 'LLama 3.3 70B',
#     '/dataset/shared_models/deepseek-r1': 'Deepseek R1 671B'
# }

display_names = {
    'common-model-small': '8B',
    'common-model': '70B',
    'deepseek-r1': '671B',
    'meta-llama/Llama-3.1-8B-Instruct': '8B',
    '/dataset/shared_models/llama3_3-70b': '70B',
    '/dataset/shared_models/deepseek-r1': '671B'
}


# Debug: Add more detailed printing
print("=== DEBUGGING PLOT ISSUES ===")

# 2) Read each folder's CSVs into a dict of lists-of-DataFrames
dfs = {}
all_dfs = {}
skip_idxs = {0, 3, 7}

for name, folder in exp_dirs.items():
    pattern = os.path.join(folder, "*.csv")
    paths   = glob.glob(pattern)
    print(f"\n{name}: Found {len(paths)} CSV files in {folder}")
    
    all_dfs[name] = [pd.read_csv(p) for p in paths]
    dfs[name] = [
        df
        for idx, df in enumerate(all_dfs[name])
        if idx not in skip_idxs
    ]
    
    print(f"{name}: After filtering, have {len(dfs[name])} DataFrames")
    for i, df in enumerate(dfs[name]):
        print(f"  DataFrame {i}: metric={df['metric'].unique()}, shape={df.shape}")
        # Check for empty dataframes
        if df.empty:
            print(f"    WARNING: DataFrame {i} is empty!")

# 3) Build dfs_all by zipping in the same provider-order
provider_order = list(exp_dirs.keys())
list_of_df_lists = [dfs[p] for p in provider_order]

# Check if all providers have the same number of files
lengths = [len(df_list) for df_list in list_of_df_lists]
print(f"\nProvider DataFrame counts: {dict(zip(provider_order, lengths))}")
if len(set(lengths)) > 1:
    print("WARNING: Not all providers have the same number of DataFrames!")

# 4) Concatenate the i-th file of each provider across all providers
dfs_all = []
for i, dfs_tuple in enumerate(zip(*list_of_df_lists)):
    print(f"\nCombining DataFrame set {i}:")
    for j, df in enumerate(dfs_tuple):
        print(f"  Provider {provider_order[j]}: {df.shape}, metric={df['metric'].unique()}")
    
    combined_df = pd.concat(dfs_tuple, ignore_index=True)
    combined_df['model'] = combined_df['model'].map(display_names).fillna(combined_df['model'])

    print(f"  Combined result: {combined_df.shape}")
    dfs_all.append(combined_df)

print(f"\nBuilt {len(dfs_all)} combined DataFrames in dfs_all")

# Remove total_tokens and accuracy - but check indices first
print(f"dfs_all length before popping: {len(dfs_all)}")
if len(dfs_all) > 4:
    total_tokens = dfs_all.pop(4)
    print("Popped index 4 (total_tokens)")
else:
    print("WARNING: Cannot pop index 4, not enough DataFrames")

if len(dfs_all) > 6:
    dpsk_output = dfs_all.pop(6)
    print("Popped index 6 (dpsk_output)")
else:
    print("WARNING: Cannot pop index 6, not enough DataFrames")

print(f"dfs_all length after popping: {len(dfs_all)}")

## TMR Table (TODO Edit)

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
    # 'input_size': df_med['input_size'],
    'median':   df_med['value'],
    'p95':      df_p95['value']
})

# 4) Compute the ratio (p99 over median)
ratio_df['p95_to_median'] = ratio_df['p95'] / ratio_df['median']

# 5) (Optional) Inspect it
print(ratio_df.head())
print(ratio_df.groupby(['provider','model'])['p95'].describe()['mean'])
print(ratio_df.groupby(['provider','model'])['median'].describe()['mean'])
print(ratio_df.groupby(['provider','model'])['p95_to_median'].describe()['mean'])

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

# 1) Group by provider AND max_output, computing the two means
summary_by_output = (
    ratio_df
    .groupby(['provider','model'], as_index=False)
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
print(summary_by_output.to_markdown(index=False))

print(ratio_df[['provider','model','p95_to_median']].head())

# Filter for desired metrics
desired_metrics = ['timetofirsttoken', 'response_times', "timebetweentokens"]
# desired_metrics = ["timebetweentokens"]
dfs_filtered = []

print("\n=== FILTERING FOR DESIRED METRICS ===")
for i, df in enumerate(dfs_all):
    if df.empty:
        print(f"DataFrame {i} is empty, skipping")
        continue
        
    metric = df['metric'].iloc[0]
    print(f"DataFrame {i}: metric='{metric}', shape={df.shape}")
    
    if metric in desired_metrics:
        print(f"  -> INCLUDED (matches desired metrics)")
        # Additional debugging for the filtered dataframe
        print(f"  -> Providers: {df['provider'].unique()}")
        print(f"  -> Models: {df['model'].unique()}")
        print(f"  -> Value column type: {type(df['value'].iloc[0])}")
        print(f"  -> Sample values: {df['value'].head(3).tolist()}")
        dfs_filtered.append(df)
    else:
        print(f"  -> SKIPPED (not in desired metrics)")

# Use filtered dataframes
dfs_all = dfs_filtered
print(f"\nFinal filtered DataFrames count: {len(dfs_all)}")

# Now create the plots with shared y-axis
display_titles = {
    'timebetweentokens': 'TBT',
    'timetofirsttoken': 'TTFT',
    'response_times': 'TRT'
}

# linestyles = {
#     'LLama 3.1 8B': 'solid',
#     'LLama 3.3 70B': 'dashed',
#     'Deepseek R1 671B': 'dashdot'
# }

linestyles = {
    '8B': 'solid',
    '70B': 'dashed',
    '671B': 'dashdot'
}

colors = {
    'vLLM': 'black',    
    'Azure': '#FF8000',
    'AWSBedrock': 'teal',
    'TogetherAI': 'purple'
}   

if len(dfs_all) == 0:
    print("ERROR: No DataFrames to plot!")
    exit()

# Create subplots with shared y-axis
fig, axes = plt.subplots(1, len(dfs_all), figsize=(13, 3.5), sharey=True)
if len(dfs_all) == 1:
    axes = [axes]  # Make it iterable

plt.rcParams.update({
    'axes.labelsize': 16,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})

# for i, df in enumerate(dfs_all):
for i, (ax, df) in enumerate(zip(axes, dfs_all)):
    print(df.head(1))
    # ax = axes[i]
    for provider in df['provider'].unique():
    # Loop over each model (not provider) and use default colors
        for model in df['model'].unique():
            print(model)
            
            # subset = df[df['model'] == model]
            subset = df[
                (df['provider'] == provider) &
                (df['model'] == model)
            ]
            if subset.empty:
                continue
            if df['metric'].iloc[0] == "response_times":
                print(subset['value'].max())
            # print("HERE", df['metric'].iloc[0])
        # choose values (convert to ms if needed)
            values = subset['value'].values
            # print(len(values))
            if df['metric'].iloc[0] == "timebetweentokens":
                flattened_values = []
                # print(len(values[0]))
                for sublist in values:
                    if isinstance(sublist, str):
                        try:
                            # Safely evaluate string as Python literal
                            parsed_list = ast.literal_eval(sublist)
                            # for x in parsed_list:
                            #     print(float(x))
                                # flattened_values.append(float(x))
                            flattened_values.extend([float(x) for x in parsed_list])
                            
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing: {sublist[:50]}... Error: {e}")
                            continue
                    else:
                        print("here?")
                        flattened_values.extend([float(x) for x in sublist])
                    
                # print(len(flattened_values), flattened_values[0], flattened_values[-1])
                values = np.array(flattened_values) * 1000
            else:
                values = subset['value'].values * 1000 

        # if df['metric'].iloc[0] != "totaltokens":
        #     vals = vals * 1000

            sorted_values = np.sort(values)
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

        # # thicker line for vLLM if you want, otherwise default line width
        #     lw = 2.2 if model == 'vLLM' else 1.5

        # # **no color=…** so default cycle is used
        # pretty = display_names.get(model, model)
        #     ax.plot(sorted_values, cdf,
        #         linewidth=lw,
        #         label=pretty)

            if provider == 'vLLM':
                pretty = display_names.get(model, model)
                ax.plot(sorted_values, cdf, color=colors[provider], linestyle=linestyles[pretty], linewidth=2.2)
            else:
                # print(model, display_names.get(model, model), linestyles[display_names.get(model, model)])
                # for size, style in linestyles.items():
                #     print(size, style)
                ax.plot(sorted_values, cdf, color=colors[provider], linestyle=linestyles[display_names.get(model, model)], linewidth=1.5)

        # Add legend only to the first subplot to avoid repetition
        # if i == 0:  # Only add legend to first subplot
            if i == (len(dfs_all) - 1):
                color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=provider) 
                                for provider, color in colors.items()]

                linestyle_legend_elements = [Line2D([0], [0], color='black', linestyle=style, lw=2, label=f"{size}") 
                                        for size, style in linestyles.items()]
            
                # ax.legend(
                #     title="Model",
                #     loc="center left",
                #     bbox_to_anchor=(1.03, 0.5),
                #     fontsize=13,
                #     title_fontsize=14,
                #     labelspacing=0.4,
                #     handlelength=0.7,
                #     frameon=True,
                #     borderpad=0.4,
                #     fancybox=False, 
                #     shadow=False, 
                #     framealpha=1.0
                # )
                # legend1 = fig.legend(
                #     handles=color_legend_elements,
                #     title='Provider',
                #     loc='center left',
                #     bbox_to_anchor=(1.01, 0.7),  # x, y coordinates (1.02 puts it outside)
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

                # legend2 = fig.legend(
                #     handles=linestyle_legend_elements, 
                #     title='Model', 
                #     loc='center left',
                #     bbox_to_anchor=(1.01, 0.3),  # Position below the first legend
                #     fontsize=13, 
                #     title_fontsize=14, 
                #     frameon=True, 
                #     borderpad=0.5,
                #     labelspacing=0.4,
                #     handlelength=2.0,
                #     handletextpad=0.5,
                #     fancybox=False, 
                #     shadow=False, 
                #     framealpha=1.0
                # )

                # Add the first legend back (matplotlib removes it when adding the second)
                
                plt.subplots_adjust(right=0.85)
            # Set labels and formatting for each subplot
            xlabel = "Latency (ms)" if df['metric'].iloc[0] != "totaltokens" else "Total Tokens"
            ax.set_xlabel(xlabel)
        
        # Only set ylabel on the leftmost subplot
            if i == 0:
                ax.set_ylabel("CDF")
        
        # Set title for each subplot
            metric_name = df['metric'].iloc[0]
            ax.set_title(display_titles.get(metric_name, metric_name))
            
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

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
    title='Model', 
    loc='center left',
    bbox_to_anchor=(0.86, 0.3),  # Position below the first legend
    fontsize=13, 
    title_fontsize=14, 
    frameon=True, 
    borderpad=0.5,
    labelspacing=0.4,
    handlelength=2.0,
    handletextpad=0.5,
    fancybox=False, 
    shadow=False, 
    framealpha=1.0
)
plt.gca().add_artist(legend1)
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Save the combined figure
ts = datetime.now().strftime("%y%m%d_%H%M")
fname = f"combined_metrics_{ts}.png"
out = os.path.join(graph_dir, fname)
plt.savefig(out)
plt.show()
plt.close()
print("Saved combined graph:", out)
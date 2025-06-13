import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.lines import Line2D
from datetime import datetime

graph_dir = "experiments/exp_20250610_122254"

azure_dir = "experiments/exp_20250606_144619"
csv_azure = glob.glob(f"{azure_dir}/*.csv")
dfs_azure = [pd.read_csv(f) for f in csv_azure]

together_dir = "experiments/exp_20250606_051340"
csv_together = glob.glob(f"{together_dir}/*.csv")
dfs_together = [pd.read_csv(f) for f in csv_together]

aws_dir_1 = "experiments/exp_20250607_170514"
csv_aws_1 = glob.glob(f"{aws_dir_1}/*.csv")
dfs_aws_1 = [pd.read_csv(f) for f in csv_aws_1]

aws_dir_2 = "experiments/exp_20250607_222813"
csv_aws_2 = glob.glob(f"{aws_dir_2}/*.csv")
dfs_aws_2 = [pd.read_csv(f) for f in csv_aws_2]

dfs_aws = [
    pd.concat([df1, df2], ignore_index=True) 
    for df1, df2 in zip(dfs_aws_1, dfs_aws_2)
]

dfs_all = [
    pd.concat([df1, df2, df_3], ignore_index=True)
    for df1, df2, df_3 in zip(dfs_together, dfs_azure, dfs_aws)
]

total_tokens = dfs_all.pop(3)

# for idx in range(len(dfs_all)):
#     print(dfs_all[idx].head(1))
#     print(dfs_all[idx].shape)
#     dfs_all[idx] = dfs_all[idx].loc[pure_exp.index]
#     print(dfs_all[idx].shape)
dfs_all = [df[df['model'] != 'common-model-small'].copy() for df in dfs_all]

# for i, df in enumerate(dfs_all):
#     print(df.head(1))
#     # want to remove entries with model = "common-model-small"
#     print(df['provider'].unique())
#     print(df['model'].unique())
#     print(df['max_output'].unique())
#     print(f"combined #{i}: {df.shape}")

for i, df in enumerate(dfs_all):
    print(df.head(1))

    plt.figure(figsize=(7, 3.5))

    # linestyles = {
    #     500: 'solid',
    #     1000: 'dashed',
    #     5000: 'dashdot',
    #     10000: ':'
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

    for provider in df['provider'].unique():
        # for max_output in sorted(df['max_output'].unique()):
        subset = df[
            (df['provider'] == provider) 
        ]
        if subset.empty:
            continue
        values = subset['value'].values * 1000  # convert to ms
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        
        # Plot without label to avoid cluttered legend
        if provider == 'vLLM':
            plt.plot(sorted_vals, cdf, color=colors[provider], linewidth=2.2)
        else:
            plt.plot(sorted_vals, cdf, color=colors[provider], linewidth=1.5)

    # Create custom legends only for the first plot (for research paper)
    if i == 0:  # Only add legends to the first plot
        # Color legend (for providers)
        color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=provider) 
                               for provider, color in colors.items()]
        
        # linestyle_legend_elements = [Line2D([0], [0], color='black', linestyle=style, lw=2, label=f'{size:,}') 
        #                            for size, style in linestyles.items()]
        
        legend1 = plt.legend(handles=color_legend_elements, title='Provider', 
                            loc='center left', bbox_to_anchor=(1.02, 0.7), 
                            fontsize=10, title_fontsize=11, frameon=True, 
                            fancybox=False, shadow=False, framealpha=1.0)
        
        # legend2 = plt.legend(handles=linestyle_legend_elements, title='Output Length', 
        #                     loc='center left', bbox_to_anchor=(1.02, 0.3), 
        #                     fontsize=10, title_fontsize=11, frameon=True,
        #                     fancybox=False, shadow=False, framealpha=1.0)
        
        # Add the first legend back (matplotlib removes it when adding the second)
        plt.gca().add_artist(legend1)

    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    
    # Smart title generation for compound words and underscores
    metric_raw = df['metric'].iloc[0]
    # Split on underscores and add spaces before capital letters in compound words
    # metric = re.sub(r'([a-z])([A-Z])', r'\1 \2', metric_raw.replace('_', ' ')).title()
    # plt.title(f"{metric} CDF by Provider and Output Length")
    if df['metric'].iloc[0] != "timebetweentokens":
        plt.xscale("log")
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to accommodate external legends only for first plot
    plt.tight_layout()
    if i == 0:
        plt.subplots_adjust(right=0.75)  # Make room for legends on the right
    plt.show()
    current_time = datetime.now().strftime("%y%m%d_%H%M")
    filename = f"{df['metric'].unique()[0]}_{current_time}.pdf"
    filepath = os.path.join(graph_dir, filename)
    plt.savefig(filepath)
    plt.close()

    print(f"Saved graph: {filepath}")

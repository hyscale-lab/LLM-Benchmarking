import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.lines import Line2D
from datetime import datetime
import re

vllm_dir_1 = "experiments/exp_20250604_045631"
csv_vllm_1 = glob.glob("experiments/exp_20250604_045631/*.csv")

vllm_dir_2 = "experiments/exp_20250604_055206"
csv_vllm_2 = glob.glob(f"{vllm_dir_2}/*.csv")

df_vllm_1 = [pd.read_csv(f) for f in csv_vllm_1]
df_vllm_2 = [pd.read_csv(f) for f in csv_vllm_2]

dfs_vllm = [
    pd.concat([df1, df2], ignore_index=True) 
    for df1, df2 in zip(df_vllm_1, df_vllm_2)
]

azure_dir = "experiments/exp_20250604_055446"
csv_azure = glob.glob(f"{azure_dir}/*.csv")
dfs_azure = [pd.read_csv(f) for f in csv_azure]

together_dir = "experiments/exp_20250604_055314"
csv_together = glob.glob(f"{together_dir}/*.csv")
dfs_together = [pd.read_csv(f) for f in csv_together]

aws_dir = "experiments/exp_20250604_085229"
csv_aws = glob.glob(f"{aws_dir}/*.csv")
dfs_aws = [pd.read_csv(f) for f in csv_aws]

dfs_all = [
    pd.concat([df1, df2, df_3, df_4], ignore_index=True)
    for df1, df2, df_3, df_4 in zip(dfs_vllm, dfs_together, dfs_azure, dfs_aws)
]

total_tokens = dfs_all.pop(3)

pure_exp = total_tokens[(total_tokens['max_output'] - total_tokens['value']).abs() <= 200]

for idx in range(len(dfs_all)):
    print(dfs_all[idx].head(1))
    print(dfs_all[idx].shape)
    dfs_all[idx] = dfs_all[idx].loc[pure_exp.index]
    print(dfs_all[idx].shape)


graph_dir = "experiments/exp_20250604_055539"

for i, df in enumerate(dfs_all):
    print(df.head(1))

    plt.figure(figsize=(7, 3.5))

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

    for provider in df['provider'].unique():
        for max_output in sorted(df['max_output'].unique()):
            subset = df[
                (df['provider'] == provider) &
                (df['max_output'] == max_output)
            ]
            if subset.empty:
                continue
            values = subset['value'].values * 1000  # convert to ms
            sorted_vals = np.sort(values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            
            # Plot without label to avoid cluttered legend
            if provider == 'vLLM':
                plt.plot(sorted_vals, cdf, color=colors[provider], linestyle=linestyles[max_output], linewidth=2.2)
            else:
                plt.plot(sorted_vals, cdf, color=colors[provider], linestyle=linestyles[max_output], linewidth=1.5)

    # Create custom legends only for the first plot (for research paper)
    if i == 0:  # Only add legends to the first plot
        # Color legend (for providers)
        color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=provider) 
                               for provider, color in colors.items()]
        
        linestyle_legend_elements = [Line2D([0], [0], color='black', linestyle=style, lw=2, label=f'{size:,}') 
                                   for size, style in linestyles.items()]
        
        legend1 = plt.legend(handles=color_legend_elements, title='Provider', 
                            loc='center left', bbox_to_anchor=(1.02, 0.7), 
                            fontsize=10, title_fontsize=11, frameon=True, 
                            fancybox=False, shadow=False, framealpha=1.0)
        
        legend2 = plt.legend(handles=linestyle_legend_elements, title='Output Length', 
                            loc='center left', bbox_to_anchor=(1.02, 0.3), 
                            fontsize=10, title_fontsize=11, frameon=True,
                            fancybox=False, shadow=False, framealpha=1.0)
        
        # Add the first legend back (matplotlib removes it when adding the second)
        plt.gca().add_artist(legend1)

    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    
    # Smart title generation for compound words and underscores
    metric_raw = df['metric'].iloc[0]
    # Split on underscores and add spaces before capital letters in compound words
    metric = re.sub(r'([a-z])([A-Z])', r'\1 \2', metric_raw.replace('_', ' ')).title()
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

# print(pure_exp.head(1))
# print(pure_exp.shape, total_tokens.shape)
# print(pure_exp['provider'].unique())
# print(pure_exp['max_output'].unique())

# for i, df in enumerate(dfs_all):
#     print(df.head(1))
#     print(df['provider'].unique())
#     print(df['max_output'].unique())
#     print(f"combined #{i}: {df.shape}")






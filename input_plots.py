import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.lines import Line2D
from datetime import datetime
import re

# graph_dir = "experiments/exp_20250604_044612"
# csv_files = glob.glob("experiments/exp_20250604_044612/*.csv")

# graph_dir = "experiments/exp_20250611_204817"
# csv_files = glob.glob("experiments/exp_20250611_204817/*.csv")

# graph_dir = "experiments/exp_20250613_122819"
# csv_files = glob.glob("experiments/exp_20250613_122819/*.csv")

graph_dir = "experiments/exp_20250616_141947_71969dba"
csv_files = glob.glob(f"{graph_dir}/*.csv")


dfs_all = [pd.read_csv(f) for f in csv_files]

# vllm_dir = "experiments/exp_20250612_224012"
# csv_vllm = glob.glob(f"{vllm_dir}/*.csv")

# dfs_vllm = [pd.read_csv(f) for f in csv_vllm]

# for i, df in enumerate(dfs_vllm):
#     print(df.head(1))
#     print(df['provider'].unique())
#     print(df['max_output'].unique())
#     print(f"combined #{i}: {df.shape}")

# dfs_all = [
#     pd.concat([df1, df2], ignore_index=True)
#     for df1, df2 in zip(df_list, dfs_vllm)
# ]

total_tokens_df = dfs_all.pop(3)

for i, df in enumerate(dfs_all):
    print(df.head(1))

    plt.figure(figsize=(7, 3.5))

    linestyles = {
        1000: 'solid',
        10000: 'dashed',
        100000: 'dashdot'
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
        for input_size in sorted(df['input_size'].unique()):
            subset = df[
                (df['provider'] == provider) &
                (df['input_size'] == input_size)
            ]
            if subset.empty:
                continue
            values = subset['value'].values * 1000  # convert to ms
            sorted_vals = np.sort(values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            
            # Plot without label to avoid cluttered legend
            if provider == 'vLLM':
                plt.plot(sorted_vals, cdf, color=colors[provider], linestyle=linestyles[input_size], linewidth=2.2)
            else:
                plt.plot(sorted_vals, cdf, color=colors[provider], linestyle=linestyles[input_size], linewidth=1.5)

    # Create custom legends only for the first plot (for research paper)
    if i == 0:  # Only add legends to the first plot
        # Color legend (for providers)
        color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=provider) 
                               for provider, color in colors.items()]
        
        # Linestyle legend (for input sizes)
        linestyle_legend_elements = [Line2D([0], [0], color='black', linestyle=style, lw=2, label=f'{size:,}') 
                                   for size, style in linestyles.items()]
        
        # Place legends outside the plot area for clean research paper presentation
        legend1 = plt.legend(handles=color_legend_elements, title='Provider', 
                            loc='center left', bbox_to_anchor=(1.02, 0.7), 
                            fontsize=10, title_fontsize=11, frameon=True, 
                            fancybox=False, shadow=False, framealpha=1.0)
        
        legend2 = plt.legend(handles=linestyle_legend_elements, title='Input Size', 
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
    # plt.title(f"{metric} CDF by Provider and Input Size")
    if df['metric'].iloc[0] != "timebetweentokens_p95" or df['metric'].iloc[0] != "timebetweentokens":
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
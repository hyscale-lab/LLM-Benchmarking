import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.lines import Line2D
from datetime import datetime
import re

df1_dir = "experiments/exp_20250611_175022"
csv_df1 = glob.glob(f"{df1_dir}/*.csv")
dfs_df1 = [pd.read_csv(f) for f in csv_df1]

anthropic_dir = "experiments/exp_20250611_215533"
csv_anthropic = glob.glob(f"{anthropic_dir}/*.csv")
dfs_anthropic = [pd.read_csv(f) for f in csv_anthropic]

vllm_dir = "experiments/exp_20250612_234701"
csv_vllm = glob.glob(f"{vllm_dir}/*.csv")
dfs_vllm = [pd.read_csv(f) for f in csv_vllm]

graph_dir = "experiments/exp_20250611_175022"

dfs_all = [
    pd.concat([df1, df2, df3], ignore_index=True)
    for df1, df2, df3 in zip(dfs_df1, dfs_anthropic, dfs_vllm)
]

total_tokens = dfs_all.pop(3)


# provider_colors = {
#     'vLLM':    'black',
#     'AWSBedrock':  'teal',
#     'Azure':     '#FF8000',
#     'Cloudflare': 'green',
#     'Open AI':  '#D22F2D',
#     'TogetherAI':  'purple',
#     'Anthropic': '#FF0095',
#     'Google': '#007FFF'
# }

for i, df in enumerate(dfs_all):
    print(df.head(1))

    plt.figure(figsize=(6.5, 4))

    # linestyles = {
    #     500: 'solid',
    #     1000: 'dashed',
    #     5000: 'dashdot',
    #     10000: ':'
    # }

    colors = {
    'vLLM':    'black',
    'AWSBedrock':  'teal',
    'Azure':     '#FF8000',
    'Cloudflare': 'green',
    'Open_AI':  '#D22F2D',
    'TogetherAI':  'purple',
    'Anthropic': '#e377c2',
    'GoogleGemini': '#007FFF'
    }

#     provider_colors = {
#     'vLLM':         '#1f77b4',  # blue
#     'AWSBedrock':   '#ff7f0e',  # orange
#     'Azure':        '#2ca02c',  # green
#     'Cloudflare':   '#d62728',  # red
#     'Open AI':      '#9467bd',  # purple
#     'TogetherAI':   '#8c564b',  # brown
#     'Anthropic':    '#e377c2',  # pink
#     'Google':       '#7f7f7f',  # gray
# }

    plt.rcParams.update({
    'axes.labelsize': 16,   # X/Y label font size
    'xtick.labelsize': 18,  # X‐tick label font size
    'ytick.labelsize': 18   # Y‐tick label font size
    })

    for provider in df['provider'].unique():
        # for max_output in sorted(df['max_output'].unique()):
            subset = df[
                (df['provider'] == provider) 
                # (df['max_output'] == max_output)
            ]
            if subset.empty:
                continue
            values = subset['value'].values * 1000  # convert to ms
            sorted_vals = np.sort(values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            
            # Plot without label to avoid cluttered legend
            if provider == 'vLLM':
                plt.plot(sorted_vals, cdf, color=colors[provider], linewidth=2.4)
            else:
                plt.plot(sorted_vals, cdf, color=colors[provider], linewidth=1.8)

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
        
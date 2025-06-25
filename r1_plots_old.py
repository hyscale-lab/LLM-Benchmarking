import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogLocator, FuncFormatter
from matplotlib.lines import Line2D
from datetime import datetime

graph_dir = "experiments/zexp_5.2_deepseek_plots/results/tokens"
# os.makedirs(graph_dir, exist_ok=True)

# azure_dir = "experiments/exp_20250606_144619"
# csv_azure = glob.glob(f"{azure_dir}/*.csv")
# dfs_azure = [pd.read_csv(f) for f in csv_azure]

# together_dir = "experiments/exp_20250606_051340"
# csv_together = glob.glob(f"{together_dir}/*.csv")
# dfs_together = [pd.read_csv(f) for f in csv_together]

# aws_dir_1 = "experiments/exp_20250607_170514"
# csv_aws_1 = glob.glob(f"{aws_dir_1}/*.csv")
# dfs_aws_1 = [pd.read_csv(f) for f in csv_aws_1]

# aws_dir_2 = "experiments/exp_20250607_222813"
# csv_aws_2 = glob.glob(f"{aws_dir_2}/*.csv")
# dfs_aws_2 = [pd.read_csv(f) for f in csv_aws_2]

# vllm_dir = "experiments/exp_20250616_223309_22c4d77e"
# csv_vllm = glob.glob(f"{vllm_dir}/*.csv")
# dfs_vllm = [pd.read_csv(f) for f in csv_vllm]

# for i, df in enumerate(dfs_vllm):
#     print(df.head(1))

# together_dir = "experiments/exp_20250618_051258_b1b4aa55"
# csv_together = glob.glob(f"{together_dir}/*.csv")
# dfs_together = [pd.read_csv(f) for f in csv_together]

# others_dir = "experiments/exp_20250617_121335_bdf9e287"
# csv_others = glob.glob(f"{others_dir}/*.csv")
# dfs_others = [pd.read_csv(f) for f in csv_others]
# dfs_others = [df[df['provider'] != 'TogetherAI'].copy() for df in dfs_others]

# # dfs_aws = [
# #     pd.concat([df1, df2], ignore_index=True) 
# #     for df1, df2 in zip(dfs_aws_1, dfs_aws_2)
# # ]

# dfs_all = [
#     pd.concat([df1, df2, df3], ignore_index=True)
#     for df1, df2, df3 in zip(dfs_vllm, dfs_together, dfs_others)
# ]

# total_tokens = dfs_all.pop(3)
# accuracy = dfs_all.pop(3)

# for i, df in enumerate(dfs_all):
#     print(df.head(1))
# #     # want to remove entries with model = "common-model-small"
#     print(df['metric'].unique())
#     print(df['provider'].unique())
#     print(df['model'].unique())
#     print(df['max_output'].unique())
#     print(f"combined #{i}: {df.shape}")

exp_dirs = {
    # "aws": "experiments/exp_20250618_162522_20f4040a",
    "aws": "experiments/exp_20250624_115804_c1ae8037",
    "azure": "experiments/exp_20250618_162137_e31b40f5",
    "vllm": "experiments/exp_20250619_112437_c70b32bc",
    "tgt": "experiments/exp_20250618_163312_f98e83f7"
}

# 2) Read each folder’s CSVs into a dict of lists-of-DataFrames
dfs = {}
all_dfs = {}
# skip_idxs = {0, 3, 7}
for name, folder in exp_dirs.items():
    print(name, folder)
    pattern = os.path.join(folder, "*.csv")
    paths   = glob.glob(pattern)
    all_dfs[name] = [pd.read_csv(p) for p in paths]
    dfs[name] = [
        df
        for idx, df in enumerate(all_dfs[name])
        # if idx not in skip_idxs
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
# total_tokens = dfs_all.pop(4)
# dpsk_output = dfs_all.pop(6)
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

colors = {
    'vLLM':       'black',
    'Azure':      '#FF8000',
    'AWSBedrock': 'teal',
    'TogetherAI': 'purple'
}

plt.rcParams.update({
    'axes.labelsize': 18,   # X/Y label font size
    'xtick.labelsize': 18,  # X‐tick label font size
    'ytick.labelsize': 18   # Y‐tick label font size
    })


plt.figure(figsize=(8, 4))
accuracy = dfs_all.pop(7)
# need to make a table for accuracy provider | accuracy
for provider in accuracy['provider'].unique():
    # print(provider)
    subset = accuracy[accuracy['provider'] == provider]
    if subset.empty:
        continue
    
    correct = 0

    for val in subset['value'].values:
        correct += val

    acc = correct/len(subset['value'].values)
    print(f"{provider}: {acc:.2f}")
    

# for provider in total_tokens['provider'].unique():
#     subset = 
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
# # filepath = os.path.join(graph_dir, filename)
# # plt.savefig(filepath)
# plt.close()

# print(f"Saved graph: {filepath}")


# for idx in range(len(dfs_all)):
#     print(dfs_all[idx].iloc[0]['metric'])
# #     print(dfs_all[idx].head(1))
# #     print(dfs_all[idx].shape)
# #     dfs_all[idx] = dfs_all[idx].loc[pure_exp.index]
# #     print(dfs_all[idx].shape)
# # dfs_all = [df[df['model'] != 'common-model-small'].copy() for df in dfs_all]

# for i, df in enumerate(dfs_all):
#     print(df.head(1))
#     # want to remove entries with model = "common-model-small"
#     print(df['provider'].unique())
#     print(df['model'].unique())
#     print(df['max_output'].unique())
#     print(f"combined #{i}: {df.shape}")

# for i, df in enumerate(dfs_all):
#     print(df.head(1))

#     plt.figure(figsize=(7, 3.5))

#     # linestyles = {
#     #     500: 'solid',
#     #     1000: 'dashed',
#     #     5000: 'dashdot',
#     #     10000: ':'
#     # }

#     colors = {
#     'vLLM': 'black',    
#     'Azure': '#FF8000',
#     'AWSBedrock': 'teal',
#     'TogetherAI': 'purple'
#     }   

#     plt.rcParams.update({
#     'axes.labelsize': 18,   # X/Y label font size
#     'xtick.labelsize': 18,  # X‐tick label font size
#     'ytick.labelsize': 18   # Y‐tick label font size
#     })

#     for provider in df['provider'].unique():
#         # for max_output in sorted(df['max_output'].unique()):
#         subset = df[
#             (df['provider'] == provider) 
#         ]
#         if subset.empty:
#             continue

#         if df['metric'].iloc[0] != "totaltokens":
#             values = subset['value'].values * 1000  # convert to ms
#         sorted_vals = np.sort(values)
#         cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        
#         # Plot without label to avoid cluttered legend
#         if provider == 'vLLM':
#             plt.plot(sorted_vals, cdf, color=colors[provider], linewidth=2.2)
#         else:
#             plt.plot(sorted_vals, cdf, color=colors[provider], linewidth=1.5)

#     # Create custom legends only for the first plot (for research paper)
#     if i == 0:  # Only add legends to the first plot
#         # Color legend (for providers)
#         color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=provider) 
#                                for provider, color in colors.items()]
        
#         # linestyle_legend_elements = [Line2D([0], [0], color='black', linestyle=style, lw=2, label=f'{size:,}') 
#         #                            for size, style in linestyles.items()]
        
#         # legend1 = plt.legend(handles=color_legend_elements, title='Provider', 
#         #                     loc='center left', bbox_to_anchor=(1.02, 0.7), 
#         #                     fontsize=10, title_fontsize=11, frameon=True, 
#         #                     fancybox=False, shadow=False, framealpha=1.0)

#         legend1 = plt.legend(
#             handles=color_legend_elements,
#             title='Provider',
#             # loc='upper left',
#             loc="best",
#             # x0, y0, width, height  (in axes‐fraction coords)
#             # bbox_to_anchor=(1.0, 0.7, 0.5, 0.3),
#             bbox_to_anchor=(1.0, 0.7),
#             fontsize=13,
#             title_fontsize=14,
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
        
#         # legend2 = plt.legend(handles=linestyle_legend_elements, title='Output Length', 
#         #                     loc='center left', bbox_to_anchor=(1.02, 0.3), 
#         #                     fontsize=10, title_fontsize=11, frameon=True,
#         #                     fancybox=False, shadow=False, framealpha=1.0)
        
#         # Add the first legend back (matplotlib removes it when adding the second)
#         plt.gca().add_artist(legend1)

#     if df['metric'].iloc[0] != "totaltokens":
#         plt.xlabel("Latency (ms)")
#     else:
#         plt.xlabel("Total Tokens")
#     plt.ylabel("CDF")
    
#     # Smart title generation for compound words and underscores
#     metric_raw = df['metric'].iloc[0]
#     # Split on underscores and add spaces before capital letters in compound words
#     # metric = re.sub(r'([a-z])([A-Z])', r'\1 \2', metric_raw.replace('_', ' ')).title()
#     # plt.title(f"{metric} CDF by Provider and Output Length")
#     if df['metric'].iloc[0] != "totaltokens":
#         plt.xscale("log")
#     plt.grid(True, alpha=0.3)
    
#     # Adjust layout to accommodate external legends only for first plot
#     plt.tight_layout()
#     # if i == 0:
#     #     plt.subplots_adjust(right=0.75)  # Make room for legends on the right
#     plt.show()
#     current_time = datetime.now().strftime("%y%m%d_%H%M")
#     filename = f"{df['metric'].unique()[0]}_{current_time}.pdf"
#     filepath = os.path.join(graph_dir, filename)
#     plt.savefig(filepath)
#     plt.close()

#     print(f"Saved graph: {filepath}")

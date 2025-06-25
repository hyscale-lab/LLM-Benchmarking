import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.lines import Line2D
from datetime import datetime

graph_dir = "experiments/zexp_5.2_modelsize_plots/results/results_together"

# azure_dir_dpsk = "experiments/exp_20250617_183833_8f80f131"
# csv_azure_dpsk = glob.glob(f"{azure_dir_dpsk}/*.csv")
# dfs_azure_dpsk = [pd.read_csv(f) for f in csv_azure_dpsk]
# dfs_azure_dpsk = [df[df['model'] != 'common-model-small'].copy() for df in dfs_azure_dpsk]
# total_tokens = dfs_azure_dpsk.pop(3)   
# accuracy = dfs_azure_dpsk.pop(3)

# azure_dir_70b = "experiments/exp_20250617_174825_899edf8b"
# csv_azure_70b = glob.glob(f"{azure_dir_70b}/*.csv")
# dfs_azure_70b = [pd.read_csv(f) for f in csv_azure_70b]
# total_tokens = dfs_azure_70b.pop(3)
# accuracy = dfs_azure_70b.pop(3)

# azure_dir_8b = "experiments/exp_20250617_170301_df6d1feb"
# csv_azure_8b = glob.glob(f"{azure_dir_8b}/*.csv")
# dfs_azure_8b = [pd.read_csv(f) for f in csv_azure_8b]
# total_tokens = dfs_azure_8b.pop(3)
# accuracy = dfs_azure_8b.pop(3)


together_dir_dpsk = "experiments/exp_20250606_051340"
csv_together_dpsk = glob.glob(f"{together_dir_dpsk}/*.csv")
dfs_together_dpsk = [pd.read_csv(f) for f in csv_together_dpsk]
dfs_together_dpsk = [df[df['model'] != 'common-model-small'].copy() for df in dfs_together_dpsk]

total_tokens = dfs_together_dpsk.pop(3)
accuracy = dfs_together_dpsk.pop(3)

# azure_dir = "experiments/exp_20250617_174825_899edf8b"
# csv_azure = glob.glob(f"{azure_dir}/*.csv")
# dfs_azure = [pd.read_csv(f) for f in csv_azure]

together_dir_70b = "experiments/exp_20250617_174639_880fbe3e"
csv_together_70b = glob.glob(f"{together_dir_70b}/*.csv")
dfs_together_70b = [pd.read_csv(f) for f in csv_together_70b]
total_tokens = dfs_together_70b.pop(3)
accuracy = dfs_together_70b.pop(3)

together_dir_8b = "experiments/exp_20250617_171300_794a57cb"
csv_together_8b = glob.glob(f"{together_dir_8b}/*.csv")
dfs_together_8b = [pd.read_csv(f) for f in csv_together_8b]
total_tokens = dfs_together_8b.pop(3)
accuracy = dfs_together_8b.pop(3)

# aws_dir_1 = "experiments/exp_20250607_170514"
# csv_aws_1 = glob.glob(f"{aws_dir_1}/*.csv")
# dfs_aws_1 = [pd.read_csv(f) for f in csv_aws_1]

# aws_dir_2 = "experiments/exp_20250607_222813"
# csv_aws_2 = glob.glob(f"{aws_dir_2}/*.csv")
# dfs_aws_2 = [pd.read_csv(f) for f in csv_aws_2]

aws_dir_dpsk = "experiments/exp_20250618_094839_615c9c22"
csv_aws_dpsk = glob.glob(f"{aws_dir_dpsk}/*.csv")
dfs_aws_dpsk = [pd.read_csv(f) for f in csv_aws_dpsk]

# dfs_aws_dpsk = [
#     pd.concat([df1, df2], ignore_index=True) 
#     for df1, df2 in zip(dfs_aws_1, dfs_aws_2)
# ]

dfs_aws_dpsk = [df[df['model'] != 'common-model-small'].copy() for df in dfs_aws_dpsk]
for i, df in enumerate(dfs_aws_dpsk):
    print(df.head(1))
#     # want to remove entries with model = "common-model-small"
    print(df['metric'].unique())
    print(df['provider'].unique())
    print(df['model'].unique())
    print(df['max_output'].unique())
    print(f"combined #{i}: {df.shape}")

total_tokens = dfs_aws_dpsk.pop(3)

# aws_dir_70b = "experiments/exp_20250617_174754_b332014b"
# csv_aws_70b = glob.glob(f"{aws_dir_70b}/*.csv")
# dfs_aws_70b = [pd.read_csv(f) for f in csv_aws_70b]
# total_tokens = dfs_aws_70b.pop(3)
# accuracy = dfs_aws_70b.pop(3)

# aws_dir_8b = "experiments/exp_20250617_170324_4cdde8b5"
# csv_aws_8b = glob.glob(f"{aws_dir_8b}/*.csv")
# dfs_aws_8b = [pd.read_csv(f) for f in csv_aws_8b]
# total_tokens = dfs_aws_8b.pop(3)
# accuracy = dfs_aws_8b.pop(3)

# dfs_all_together = [
#     pd.concat([df1, df2, df3], ignore_index=True)
#     for df1, df2, df3 in zip(dfs_together_8b, dfs_together_70b, dfs_together_dpsk)
# ] 

# dfs_all_azure = [
#     pd.concat([df1, df2, df3], ignore_index=True)
#     for df1, df2, df3 in zip(dfs_azure_8b, dfs_azure_70b, dfs_azure_dpsk)
# ] 

# dfs_all_aws = [
#     pd.concat([df1, df2, df3], ignore_index=True)
#     for df1, df2, df3 in zip(dfs_aws_8b, dfs_aws_70b, dfs_aws_dpsk)
# ] 

# # for idx in range(len(dfs_all)):
# #     print(dfs_all[idx].iloc[0]['metric'])
# #     print(dfs_all[idx].head(1))
# #     print(dfs_all[idx].shape)
# #     dfs_all[idx] = dfs_all[idx].loc[pure_exp.index]
# #     print(dfs_all[idx].shape)

# for i, df in enumerate(dfs_all_together):
#     print(df.head(1))
# #     # want to remove entries with model = "common-model-small"
#     print(df['metric'].unique())
#     print(df['provider'].unique())
#     print(df['model'].unique())
#     print(df['max_output'].unique())
#     print(f"combined #{i}: {df.shape}")

# display_names = {
#     'common-model-small':        'Meta LLama 3.1 8B',
#     'common-model':  'Meta LLama 3.3 70B',
#     'deepseek-r1':       'Deepseek R1 671B'
# }

# for i, df in enumerate(dfs_all_together):
#     print(df.head(1))
#     plt.figure(figsize=(6, 3.5))
#     plt.rcParams.update({
#         'axes.labelsize': 18,
#         'xtick.labelsize': 18,
#         'ytick.labelsize': 18
#     })

#     # Loop over each model (not provider) and use default colors
#     for model in df['model'].unique():
#         subset = df[df['model'] == model]
#         if subset.empty:
#             continue

#         # choose values (convert to ms if needed)
#         vals = subset['value'].values
#         if df['metric'].iloc[0] != "totaltokens":
#             vals = vals * 1000

#         sorted_vals = np.sort(vals)
#         cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

#         # thicker line for vLLM if you want, otherwise default line width
#         lw = 2.2 if model == 'vLLM' else 1.5

#         # **no color=…** so default cycle is used
#         pretty = display_names.get(model, model)
#         plt.plot(sorted_vals, cdf,
#                  linewidth=lw,
#                  label=pretty)

#     # simple legend inside plot
#     if df['metric'].iloc[0] == "timetofirsttoken":
#         plt.legend(
#             title="Model",
#             loc="best",
#             fontsize=14,
#             title_fontsize=14,
#             frameon=True,
#             borderpad=0.5
#         )

#     # axes, grid, save...
#     xlabel = "Latency (ms)" if df['metric'].iloc[0] != "totaltokens" else "Total Tokens"
#     plt.xlabel(xlabel)
#     plt.ylabel("CDF")
#     if df['metric'].iloc[0] != "totaltokens":
#         plt.xscale("log")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     ts = datetime.now().strftime("%y%m%d_%H%M")
#     fname = f"{df['metric'].unique()[0]}_{ts}.pdf"
#     out = os.path.join(graph_dir, fname)
#     plt.savefig(out)
#     plt.show()
#     plt.close()
#     print("Saved graph:", out)

# ---------
# for i, df in enumerate(dfs_all_together):
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

#     # for provider in df['provider'].unique():
#     for model in df['model'].unique():
#         # for max_output in sorted(df['max_output'].unique()):
#         subset = df[
#             (df['model'] == model) 
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
#     filename = f"{df['metric'].unique()[0]}_{current_time}.png"
#     filepath = os.path.join(graph_dir, filename)
#     plt.savefig(filepath)
#     plt.close()

#     print(f"Saved graph: {filepath}")


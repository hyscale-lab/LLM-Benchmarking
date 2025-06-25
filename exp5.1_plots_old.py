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

# Your existing data loading code...
# google_dir = "experiments/exp_20250618_164751_262ed75f"
# csv_google = glob.glob(f"{google_dir}/*.csv")
# dfs_google = [pd.read_csv(f) for f in csv_google]

# openai_dir = "experiments/exp_20250618_164830_e51d23bd"
# csv_openai = glob.glob(f"{openai_dir}/*.csv")
# dfs_openai = [pd.read_csv(f) for f in csv_openai]

# anthropic_dir = "experiments/exp_20250618_175801_5c145a82"
# csv_anthropic = glob.glob(f"{anthropic_dir}/*.csv")
# dfs_anthropic = [pd.read_csv(f) for f in csv_anthropic]

# azure_dir = "experiments/exp_20250618_164345_f0bbdb78"
# csv_azure = glob.glob(f"{azure_dir}/*.csv")
# dfs_azure = [pd.read_csv(f) for f in csv_azure]

# aws_dir = "experiments/exp_20250618_164129_04a19ed7"
# csv_aws = glob.glob(f"{aws_dir}/*.csv")
# dfs_aws = [pd.read_csv(f) for f in csv_aws]

# togetherai_dir = "experiments/exp_20250618_184735_750e2005"
# csv_togetherai = glob.glob(f"{togetherai_dir}/*.csv")
# dfs_togetherai = [pd.read_csv(f) for f in csv_togetherai]

# cloudflare_dir = "experiments/exp_20250618_165521_fcdfe7ed"
# csv_cloudflare = glob.glob(f"{cloudflare_dir}/*.csv")
# dfs_cloudlare = [pd.read_csv(f) for f in csv_cloudflare]

# vllm_dir = "experiments/exp_20250618_165035_d0f540aa"
# csv_vllm = glob.glob(f"{vllm_dir}/*.csv")
# dfs_vllm = [pd.read_csv(f) for f in csv_vllm]

graph_dir = "experiments/zexp_5.1_plots"

# dfs_all = [
#     pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
#     for df1, df2, df3, df4, df5, df6, df7, df8 in zip(dfs_google, dfs_openai, dfs_anthropic, dfs_azure, dfs_aws, dfs_togetherai, dfs_cloudlare, dfs_vllm)
# ]

import glob, os
import pandas as pd

# 1) Map each “provider” key to its experiment folder
exp_dirs = {
    "google":      "experiments/exp_20250618_164751_262ed75f",
    "openai":      "experiments/exp_20250618_164830_e51d23bd",
    "anthropic":   "experiments/exp_20250619_192224_1a594ae5",
    "azure":       "experiments/exp_20250618_164345_f0bbdb78",
    "aws":         "experiments/exp_20250618_164129_04a19ed7",
    "togetherai":  "experiments/exp_20250618_184735_750e2005",
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
        print(i, df['metric'].unique())
    # i want to remove i = 0,3,7
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
    print(f"combined #{i}: {df.shape}")
    print("-----------------")

accuracy = dfs_all.pop(3)


# Filter to only include the desired metrics
desired_metrics = ['timetofirsttoken', 'response_times', "timebetweentokens"]
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
fig, axes = plt.subplots(1, len(dfs_all), figsize=(13, 5), sharey=True)

# Set global font sizes
plt.rcParams.update({
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 18
})

# Plot each metric in its own subplot
for i, (ax, df) in enumerate(zip(axes, dfs_all)):
    print(f"Processing subplot {i}: {df['metric'].iloc[0]}")
    
    for provider in df['provider'].unique():
        subset = df[df['provider'] == provider]
        if subset.empty:
            continue
        
        # values = subset['value'].values * 1000  # convert to ms
        # if df['metric'].iloc[0] == "timebetweentokens":
        #     print(len(subset['value'].values[0]), len(subset['value'].values)) # ex output for this line - 2096 100; so there it is a list of 100 lists 
        #     continue
        # sorted_vals = np.sort(values)
        # cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        values = subset['value'].values

        # if df['metric'].iloc[0] == "timebetweentokens":
        #     print(f"Number of sublists: {len(values)}")
        #     print(f"Length of first sublist: {len(values[0])}")
        #     print(f"Sample values: {values[0][:10]}")  # Debug: see what the values look like
            
        #     # Flatten the nested lists and convert to float
        #     flattened_values = []
        #     for sublist in values:
        #         for val in sublist:
        #             print(val)
        #             flattened_values.append(float(val))  # Convert string to float
            
        #     # Convert to numpy array and then to milliseconds
        #     values = np.array(flattened_values) * 1000
            
        # else:
        #     # For non-nested data, convert to float first
        #     values = np.array([float(x) for x in values]) * 1000

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

        # Plot with different line widths for vLLM
        if provider == 'vLLM':
            ax.plot(sorted_vals, cdf, color=colors[provider], linewidth=2.4, label=provider)
        else:
            ax.plot(sorted_vals, cdf, color=colors[provider], linewidth=1.8, label=provider)
    
    # Set x-axis to log scale if not timebetweentokens
    # if df['metric'].iloc[0] != "timebetweentokens":
    ax.set_xscale("log")
    
    # Generate clean title
    metric_raw = df['metric'].iloc[0]
    metric = re.sub(r'([a-z])([A-Z])', r'\1 \2', metric_raw.replace('_', ' ')).title()
    ax.set_title(metric, fontsize=16, pad=20)
    
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
axes[0].set_ylim(0, 1.02)

# Create legend outside the plot area
# Get handles and labels from the first subplot (they should be the same for all)
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
        
legend1 = plt.legend(
    handles=color_legend_elements,
    title='Provider',
    # loc='upper left',
    loc="best",
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
# plt.subplots_adjust(top=0.85)  # Make room for legend at top

# Save the combined plot
current_time = datetime.now().strftime("%y%m%d_%H%M")
filename = f"selected_metrics_{current_time}.pdf"
filepath = os.path.join(graph_dir, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved selected metrics graph: {filepath}")
print(f"Included metrics: {[df['metric'].iloc[0] for df in dfs_all]}")

# ---------

# import pandas as pd
# import glob
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import ScalarFormatter, LogLocator
# from matplotlib.lines import Line2D
# from datetime import datetime
# import re

# # df1_dir = "experiments/exp_20250611_175022"
# # csv_df1 = glob.glob(f"{df1_dir}/*.csv")
# # dfs_df1 = [pd.read_csv(f) for f in csv_df1]

# # anthropic_dir = "experiments/exp_20250611_215533"
# # csv_anthropic = glob.glob(f"{anthropic_dir}/*.csv")
# # dfs_anthropic = [pd.read_csv(f) for f in csv_anthropic]

# # vllm_dir = "experiments/exp_20250612_234701"
# # csv_vllm = glob.glob(f"{vllm_dir}/*.csv")
# # dfs_vllm = [pd.read_csv(f) for f in csv_vllm]

# google_dir = "experiments/exp_20250617_165546_65f9e12f"
# csv_google = glob.glob(f"{google_dir}/*.csv")
# dfs_google = [pd.read_csv(f) for f in csv_google]

# openai_dir = "experiments/exp_20250617_170202_639f18be"
# csv_openai = glob.glob(f"{openai_dir}/*.csv")
# dfs_openai = [pd.read_csv(f) for f in csv_openai]

# anthropic_dir = "experiments/exp_20250617_170037_99af90fe"
# csv_anthropic = glob.glob(f"{anthropic_dir}/*.csv")
# dfs_anthropic = [pd.read_csv(f) for f in csv_anthropic]

# azure_dir = "experiments/exp_20250617_170301_df6d1feb"
# csv_azure = glob.glob(f"{azure_dir}/*.csv")
# dfs_azure = [pd.read_csv(f) for f in csv_azure]

# aws_dir = "experiments/exp_20250617_170324_4cdde8b5"
# csv_aws = glob.glob(f"{aws_dir}/*.csv")
# dfs_aws = [pd.read_csv(f) for f in csv_aws]

# togetherai_dir = "experiments/exp_20250617_171300_794a57cb"
# csv_togetherai = glob.glob(f"{togetherai_dir}/*.csv")
# dfs_togetherai = [pd.read_csv(f) for f in csv_togetherai]

# cloudflare_dir = "experiments/exp_20250617_165722_5794c09e"
# csv_cloudflare = glob.glob(f"{cloudflare_dir}/*.csv")
# dfs_cloudlare = [pd.read_csv(f) for f in csv_cloudflare]

# vllm_dir = "experiments/exp_20250617_170227_15dc44bd"
# csv_vllm = glob.glob(f"{vllm_dir}/*.csv")
# dfs_vllm = [pd.read_csv(f) for f in csv_vllm]

# graph_dir = "experiments/zexp_5.1_plots"

# # for dfs in [dfs_google, dfs_openai, dfs_anthropic, dfs_azure, dfs_aws, dfs_togetherai, dfs_cloudlare, dfs_vllm]:
# #     for i, df in enumerate(dfs):
# #         print(df.head(1))

# dfs_all = [
#     pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
#     for df1, df2, df3, df4, df5, df6, df7, df8 in zip(dfs_google, dfs_openai, dfs_anthropic, dfs_azure, dfs_aws, dfs_togetherai, dfs_cloudlare, dfs_vllm)
# ]

# # print(dfs_all)

# # for i, df in enumerate(dfs_all):
# #     print(df.head(1))
# #     print(df['provider'].unique())
# #     print(df['max_output'].unique())
# #     print(f"combined #{i}: {df.shape}")

# total_tokens = dfs_all.pop(3)
# accuracy = dfs_all.pop(3)

# # for i, df in enumerate(dfs_all):
# #     print(df.head(1)['metric'])
# #     # print(df['provider'].unique())
# #     # print(df['max_output'].unique())
# #     # print(f"combined #{i}: {df.shape}")

# # provider_colors = {
# #     'vLLM':    'black',
# #     'AWSBedrock':  'teal',
# #     'Azure':     '#FF8000',
# #     'Cloudflare': 'green',
# #     'Open AI':  '#D22F2D',
# #     'TogetherAI':  'purple',
# #     'Anthropic': '#FF0095',
# #     'Google': '#007FFF'
# # }

# for i, df in enumerate(dfs_all):
#     print(i, df.head(1))

#     plt.figure(figsize=(6.5, 4))

#     # linestyles = {
#     #     500: 'solid',
#     #     1000: 'dashed',
#     #     5000: 'dashdot',
#     #     10000: ':'
#     # }

#     colors = {
#     'vLLM':    'black',
#     'AWSBedrock':  'teal',
#     'Azure':     '#FF8000',
#     'Cloudflare': 'green',
#     'Open_AI':  '#D22F2D',
#     'TogetherAI':  'purple',
#     'Anthropic': '#e377c2',
#     'GoogleGemini': '#007FFF'
#     }

# #     provider_colors = {
# #     'vLLM':         '#1f77b4',  # blue
# #     'AWSBedrock':   '#ff7f0e',  # orange
# #     'Azure':        '#2ca02c',  # green
# #     'Cloudflare':   '#d62728',  # red
# #     'Open AI':      '#9467bd',  # purple
# #     'TogetherAI':   '#8c564b',  # brown
# #     'Anthropic':    '#e377c2',  # pink
# #     'Google':       '#7f7f7f',  # gray
# # }

#     plt.rcParams.update({
#     'axes.labelsize': 18,   # X/Y label font size
#     'xtick.labelsize': 18,  # X‐tick label font size
#     'ytick.labelsize': 18   # Y‐tick label font size
#     })

#     for provider in df['provider'].unique():
#         # for max_output in sorted(df['max_output'].unique()):
#             subset = df[
#                 (df['provider'] == provider) 
#                 # (df['max_output'] == max_output)
#             ]
#             if subset.empty:
#                 continue
#             values = subset['value'].values * 1000  # convert to ms
#             sorted_vals = np.sort(values)
#             cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            
#             # Plot without label to avoid cluttered legend
#             if provider == 'vLLM':
#                 plt.plot(sorted_vals, cdf, color=colors[provider], linewidth=2.4)
#             else:
#                 plt.plot(sorted_vals, cdf, color=colors[provider], linewidth=1.8)

#     # Create custom legends only for the first plot (for research paper)
#     if df['metric'].unique()[0] == "timebetweentokens_p95":  # Only add legends to the first plot
#         # Color legend (for providers)
#         print("HELLOO")
#         print("----------")
#         color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=provider) 
#                                for provider, color in colors.items()]
        
#         # linestyle_legend_elements = [Line2D([0], [0], color='black', linestyle=style, lw=2, label=f'{size:,}') 
#         #                            for size, style in linestyles.items()]
        
#         plt.ylim(0, 1)  
#         legend1 = plt.legend(
#             handles=color_legend_elements,
#             title='Provider',
#             # loc='upper left',
#             loc="best",
#             # x0, y0, width, height  (in axes‐fraction coords)
#             # bbox_to_anchor=(1.0, 0.7, 0.5, 0.3),
#             bbox_to_anchor=(0.4, 0.98),
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
#     # if df['metric'].unique()[0] == "timebetweentokens_p95":
#     #     plt.subplots_adjust(right=0.68)  # Make room for legends on the right
#     plt.show()
#     current_time = datetime.now().strftime("%y%m%d_%H%M")
#     filename = f"{df['metric'].unique()[0]}_{current_time}.png"
#     filepath = os.path.join(graph_dir, filename)
#     plt.savefig(filepath)
#     plt.close()

#     print(f"Saved graph: {filepath}")
        
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
#python plot_data.py /scratch/project_2007095/attieh/All_distillation_methods/logs/distill.txt ./output.png
def parse_log_file(filename, mode):
    data = []
    pattern = re.compile(r' \| '+ mode +r' \| (\{.*\})')

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                json_str = match.group(1)
                metrics = json.loads(json_str)
                data.append(metrics)

    df = pd.DataFrame(data)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df


def plot_with_two_yaxes(ax, x, y1, y2, y1_label, y2_label, title):
    ax.plot(x, y1, 'o-', color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y1_label, color='blue')
    ax.set_title(title)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True)
    
    ax2 = ax.twinx()  # Create second y-axis
    ax2.plot(x, y2, 'x-', color='red')
    ax2.set_ylabel(y2_label, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

def main(log_file_path, output_image_path):
    # Parse the logs
    train_df = parse_log_file(log_file_path, 'train')
    valid_df = parse_log_file(log_file_path, 'valid')

    # Merge on 'epoch'
    merged_df = pd.merge(train_df, valid_df, on='epoch', suffixes=('_train', '_valid'))
    
    variables_wkd = [
        [['train_loss', 'valid_loss'], ['train_kd_loss', 'train_mle_loss']],
        [["train_ppl", "valid_ppl"], ["valid_kd_loss", "valid_mle_loss"]],
        [['valid_loss', 'valid_bleu'], ['train_lr', 'train_gnorm']],
        [["train_clip", "train_loss_scale"], ["train_train_wall", "train_gb_free"]]
    ]

    variables_skd = [
        [['train_loss', 'valid_loss'], ["train_ppl", "valid_ppl"], ],
        [['valid_loss', 'valid_bleu'], ['train_lr', 'train_gnorm']],
        [["train_clip", "train_loss_scale"], ["train_train_wall", "train_gb_free"]]
    ]

    variables = np.array(variables_skd)

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(variables.shape[0], variables.shape[1], figsize=(20, 25))
    fig.suptitle('Training and Validation Metrics Overview', fontsize=20)

    for i in range(variables.shape[0]):
        for j in range(variables.shape[1]):
            plot_with_two_yaxes(axs[i, j], valid_df['epoch'], merged_df[variables[i][j][0]], merged_df[variables[i][j][1]], 
                                variables[i][j][0], variables[i][j][1], ' vs '.join(variables[i][j]))

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_image_path)
    plt.show()

if __name__ == "__main__":
    # Command-line arguments for log file path and output image path
    parser = argparse.ArgumentParser(description="Parse log files and plot metrics.")
    parser.add_argument('log_file_path', type=str, help='Path to the log file')
    parser.add_argument('output_image_path', type=str, help='Path to save the output image as .png')

    args = parser.parse_args()
    
    main(args.log_file_path, args.output_image_path)


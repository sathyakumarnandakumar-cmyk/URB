import matplotlib.pyplot as plt
from PIL import Image
import os

def combine_plots():
    bandit_path = "results/bandit_long_run_42/plots/travel_times.png"
    iql_path = "results/iql_ing_4/plots/travel_times.png"
    output_path = "results/combined_travel_times.png"
    
    if not os.path.exists(bandit_path):
        print(f"Error: {bandit_path} does not exist.")
        return
    if not os.path.exists(iql_path):
        print(f"Error: {iql_path} does not exist.")
        return

    # Load images
    img_bandit = Image.open(bandit_path)
    img_iql = Image.open(iql_path)

    # Create a figure with 2 subplots (side by side)
    # We'll make it quite wide to fit both comfortably
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Display Bandit REINFORCE
    axes[0].imshow(img_bandit)
    axes[0].axis('off')  # Hide axes
    axes[0].set_title("Bandit REINFORCE (bandit_long_run_42)\nStable Convergence", fontsize=14, fontweight='bold')

    # Display IQL
    axes[1].imshow(img_iql)
    axes[1].axis('off')  # Hide axes
    axes[1].set_title("Independent Q-Learning (iql_ing_4)\nHigh Variance & Herding", fontsize=14, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Successfully combined plots into: {output_path}")

if __name__ == "__main__":
    combine_plots()

import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

def combine_all_plots(exp1_path, exp2_path, output_dir, exp1_title, exp2_title):
    os.makedirs(output_dir, exist_ok=True)
    
    # We want to combine plots from the main 'plots' folder
    exp1_plots_dir = os.path.join(exp1_path, 'plots')
    exp2_plots_dir = os.path.join(exp2_path, 'plots')
    
    if not os.path.exists(exp1_plots_dir):
        print(f"Error: {exp1_plots_dir} does not exist.")
        return
    if not os.path.exists(exp2_plots_dir):
        print(f"Error: {exp2_plots_dir} does not exist.")
        return
        
    # Find all common PNG files
    exp1_files = {os.path.basename(f) for f in glob.glob(os.path.join(exp1_plots_dir, "*.png"))}
    exp2_files = {os.path.basename(f) for f in glob.glob(os.path.join(exp2_plots_dir, "*.png"))}
    
    common_files = exp1_files.intersection(exp2_files)
    
    if not common_files:
        print("No common plots found to combine.")
        return
        
    print(f"Found common plots to combine: {common_files}")
    
    for filename in common_files:
        img1_path = os.path.join(exp1_plots_dir, filename)
        img2_path = os.path.join(exp2_plots_dir, filename)
        
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        axes[0].imshow(img1)
        axes[0].axis('off')
        axes[0].set_title(exp1_title, fontsize=16, fontweight='bold', pad=20)
        
        axes[1].imshow(img2)
        axes[1].axis('off')
        axes[1].set_title(exp2_title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        out_file = os.path.join(output_dir, f"combined_{filename}")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved combined plot: {out_file}")

if __name__ == "__main__":
    exp1 = "results/bandit_long_run_42"
    exp2 = "results/iql_ing_4"
    out_dir = "results/combined_comparison_plots"
    
    combine_all_plots(
        exp1_path=exp1, 
        exp2_path=exp2, 
        output_dir=out_dir, 
        exp1_title="Bandit REINFORCE (bandit_long_run_42)", 
        exp2_title="Independent Q-Learning (iql_ing_4)"
    )

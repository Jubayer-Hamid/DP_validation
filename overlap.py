# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load data from a .pkl file
# def load_pkl_file(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     return data

# # Process per_sample_losses to average and find indices outside the top 40% quantile
# def process_losses(data):
#     per_sample_losses = data['per_sample_losses']  # List of 10 tensors of shape (256, 24)
    
#     # Step 1: Average loss for each sample
#     losses = [np.mean(tensor, axis=1) for tensor in per_sample_losses]  # Shape becomes (256, 1) for each batch
#     avg_losses = np.concatenate(losses)  # Concatenate to form shape (2560, 1)
    
#     # Step 2: Find indices of losses outside top 40% quantile
#     quantile_value = np.percentile(avg_losses, 60)  # Top 40% quantile (keeping 60% as the threshold)
#     indices = np.where(avg_losses < quantile_value)[0]  # Indices of samples outside top 40%
    
#     return indices

# # Generate heatmap of overlap between stored indices
# def generate_heatmap(overlap_matrix, output_dir):
#     plt.figure(figsize=(10, 8))
    
#     # Set vmin and vmax to control the color gradient
#     sns.heatmap(overlap_matrix, cmap="plasma", cbar=True, square=True, vmin=np.min(overlap_matrix), vmax=np.max(overlap_matrix))
    
#     plt.title('Overlap of Indices Outside Top 40% Quantile Across Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Epochs')
#     plt.tight_layout()
#     plt.savefig(os.path.join('/iris/u/jubayer/diffusion_policy', 'overlap_heatmap.png'))
#     plt.show()

# # Main function to process all epochs and generate heatmap
# def main(output_dir, start_epoch=0, end_epoch=450):
#     all_indices = []

#     # Process each epoch
#     for epoch in range(start_epoch, end_epoch + 1):
#         file_path = os.path.join(output_dir, f'validation_data_epoch_{epoch}.pkl')
        
#         # Load and process the epoch's losses
#         data = load_pkl_file(file_path)
#         indices = process_losses(data)
#         all_indices.append(indices)

#     # Step 3: Calculate overlap between epochs
#     num_epochs = len(all_indices)
#     overlap_matrix = np.zeros((num_epochs, num_epochs))

#     for i in range(num_epochs):
#         for j in range(num_epochs):
#             # Compute intersection of indices
#             intersection = len(np.intersect1d(all_indices[i], all_indices[j]))
#             overlap_matrix[i, j] = intersection

#     # Step 4: Generate and save the heatmap
#     generate_heatmap(overlap_matrix, output_dir)

# # Example usage
# # output_dir = "/iris/u/jubayer/diffusion_policy/data/outputs/2024.09.11/10.04.41_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"  # Set your actual output directory path
# output_dir = "/iris/u/jubayer/diffusion_policy/data/outputs/2024.09.12/07.32.23_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"  # Set your actual output directory path

# main(output_dir, start_epoch=0, end_epoch=286)

##################################################################################################################################################################################################################

import os
import pickle
import numpy as np

# Load data from a .pkl file
def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Process per_sample_losses to average the losses across epochs for each datapoint
def process_losses_across_epochs(output_dir, start_epoch, end_epoch):
    all_losses = []

    # Load and process losses for each epoch
    for epoch in range(start_epoch, end_epoch + 1):
        file_path = os.path.join(output_dir, f'validation_data_epoch_{epoch}.pkl')
        
        # Load the .pkl data
        data = load_pkl_file(file_path)
        
        # Extract and average per_sample_losses for each sample at this epoch
        per_sample_losses = data['per_sample_losses']  # List of 10 tensors of shape (256, 24)
        losses = [np.mean(tensor, axis=1) for tensor in per_sample_losses]  # Shape becomes (256, 1) for each batch
        avg_losses = np.concatenate(losses)  # Shape becomes (2560, 1)
        
        all_losses.append(avg_losses)

    # Stack all the losses across epochs to get shape (num_epochs, 2560)
    all_losses = np.stack(all_losses)  # Shape (num_epochs, 2560)

    # Step 1: Calculate the mean loss for each datapoint across all epochs
    mean_losses = np.mean(all_losses, axis=0)  # Shape (2560,)

    return mean_losses

# Find indices of datapoints that have the bottom 40% of average loss
def get_bottom_40_percent_indices(mean_losses):
    
    # Calculate the threshold for the bottom 40%
    threshold = np.percentile(mean_losses, 40)
    
    # Get the indices of datapoints with loss less than the threshold
    bottom_40_indices = np.where(mean_losses < threshold)[0]

    # find the indices of the datapoints below 5% of the losses
    threshold = np.percentile(mean_losses, 5)
    bottom_5_indices = np.where(mean_losses <= threshold)[0]
    print(f"Bottom 5% of datapoints based on average validation loss: {bottom_5_indices.tolist()}")

    # find the indices of the datapoints above 5% and below 10% of the losses
    threshold = np.percentile(mean_losses, 10)
    threshold2 = np.percentile(mean_losses, 5)
    bottom_10_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 10% of datapoints based on average validation loss: {bottom_10_indices.tolist()}")


    # find the indices of the datapoints above 10% and below 15% of the losses
    threshold = np.percentile(mean_losses, 15)
    threshold2 = np.percentile(mean_losses, 10)
    bottom_15_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 15% of datapoints based on average validation loss: {bottom_15_indices.tolist()}")

    # find the indices of the datapoints above 15% and below 20% of the losses
    threshold = np.percentile(mean_losses, 20)
    threshold2 = np.percentile(mean_losses, 15)
    bottom_20_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 20% of datapoints based on average validation loss: {bottom_20_indices.tolist()}")

    # find the indices of the datapoints above 20% and below 25% of the losses
    threshold = np.percentile(mean_losses, 25)
    threshold2 = np.percentile(mean_losses, 20)
    bottom_25_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 25% of datapoints based on average validation loss: {bottom_25_indices.tolist()}")

    # find the indices of the datapoints above 25% and below 30% of the losses
    threshold = np.percentile(mean_losses, 30)
    threshold2 = np.percentile(mean_losses, 25)
    bottom_30_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 30% of datapoints based on average validation loss: {bottom_30_indices.tolist()}")

    # find the indices of the datapoints above 30% and below 35% of the losses
    threshold = np.percentile(mean_losses, 35)
    threshold2 = np.percentile(mean_losses, 30)
    bottom_35_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 35% of datapoints based on average validation loss: {bottom_35_indices.tolist()}")

    # find the indices of the datapoints above 35% and below 40% of the losses
    threshold = np.percentile(mean_losses, 40)
    threshold2 = np.percentile(mean_losses, 35)
    bottom_40_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 40% of datapoints based on average validation loss: {bottom_40_indices.tolist()}")

    # find the indices of the datapoints above 40% and below 45% of the losses
    threshold = np.percentile(mean_losses, 45)
    threshold2 = np.percentile(mean_losses, 40)
    bottom_45_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 45% of datapoints based on average validation loss: {bottom_45_indices.tolist()}")

    # find the indices of the datapoints above 45% and below 50% of the losses
    threshold = np.percentile(mean_losses, 50)
    threshold2 = np.percentile(mean_losses, 45)
    bottom_50_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 50% of datapoints based on average validation loss: {bottom_50_indices.tolist()}")

    # find the indices of the datapoints above 50% and below 55% of the losses
    threshold = np.percentile(mean_losses, 55)
    threshold2 = np.percentile(mean_losses, 50)
    bottom_55_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 55% of datapoints based on average validation loss: {bottom_55_indices.tolist()}")

    # find the indices of the datapoints above 55% and below 60% of the losses
    threshold = np.percentile(mean_losses, 60)
    threshold2 = np.percentile(mean_losses, 55)
    bottom_60_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 60% of datapoints based on average validation loss: {bottom_60_indices.tolist()}")

    # find the indices of the datapoints above 60% and below 65% of the losses
    threshold = np.percentile(mean_losses, 65)
    threshold2 = np.percentile(mean_losses, 60)
    bottom_65_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 65% of datapoints based on average validation loss: {bottom_65_indices.tolist()}")

    # find the indices of the datapoints above 65% and below 70% of the losses
    threshold = np.percentile(mean_losses, 70)
    threshold2 = np.percentile(mean_losses, 65)
    bottom_70_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 70% of datapoints based on average validation loss: {bottom_70_indices.tolist()}")

    # find the indices of the datapoints above 70% and below 75% of the losses
    threshold = np.percentile(mean_losses, 75)
    threshold2 = np.percentile(mean_losses, 70)
    bottom_75_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 75% of datapoints based on average validation loss: {bottom_75_indices.tolist()}")

    # find the indices of the datapoints above 75% and below 80% of the losses
    threshold = np.percentile(mean_losses, 80)
    threshold2 = np.percentile(mean_losses, 75)
    bottom_80_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 80% of datapoints based on average validation loss: {bottom_80_indices.tolist()}")

    # find the indices of the datapoints above 80% and below 85% of the losses
    threshold = np.percentile(mean_losses, 85)
    threshold2 = np.percentile(mean_losses, 80)
    bottom_85_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 85% of datapoints based on average validation loss: {bottom_85_indices.tolist()}")

    # find the indices of the datapoints above 85% and below 90% of the losses
    threshold = np.percentile(mean_losses, 90)
    threshold2 = np.percentile(mean_losses, 85)
    bottom_90_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 90% of datapoints based on average validation loss: {bottom_90_indices.tolist()}")

    # find the indices of the datapoints above 90% and below 95% of the losses
    threshold = np.percentile(mean_losses, 95)
    threshold2 = np.percentile(mean_losses, 90)
    bottom_95_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 95% of datapoints based on average validation loss: {bottom_95_indices.tolist()}")

    # find the indices of the datapoints above 95% and below 100% of the losses
    threshold = np.percentile(mean_losses, 100)
    threshold2 = np.percentile(mean_losses, 95)
    bottom_100_indices = np.where((mean_losses <= threshold) & (mean_losses > threshold2))[0]
    print(f"Bottom 100% of datapoints based on average validation loss: {bottom_100_indices.tolist()}")


    
    
    return bottom_40_indices.tolist()  # Return as a list

# Main function to process all epochs, calculate mean losses, and return bottom 40% indices as a list
def main(output_dir, start_epoch=0, end_epoch=286):
    # Step 1: Process the losses across all epochs and calculate mean losses per datapoint
    mean_losses = process_losses_across_epochs(output_dir, start_epoch, end_epoch)
    
    # Step 2: Find indices of datapoints in the bottom 40% of mean losses
    bottom_40_indices = get_bottom_40_percent_indices(mean_losses)
    
    # print(f"Bottom 40% of datapoints based on average validation loss: {bottom_40_indices}")
    
    return bottom_40_indices

# Example usage
# output_dir = "/iris/u/jubayer/diffusion_policy/data/outputs/2024.09.11/10.04.41_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"  # Set your actual output directory path
output_dir = "/iris/u/jubayer/diffusion_policy/data/outputs/2024.09.12/07.32.23_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"  # Set your actual output directory path
# output_dir = "/iris/u/jubayer/diffusion_policy/data/outputs/2024.09.17/03.22.10_train_diffusion_unet_lowdim_pusht_lowdim"  # Example directory

# main(output_dir, start_epoch=0, end_epoch=3300)
main(output_dir, start_epoch=0, end_epoch=4999)

#########################################################################################################################################################

# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter

# # Load data from a .pkl file
# def load_pkl_file(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     return data

# # Process per_sample_losses to average the losses across epochs for each datapoint
# def process_losses_across_epochs(output_dir, start_epoch, end_epoch):
#     all_losses = []

#     # Load and process losses for each epoch
#     for epoch in range(start_epoch, end_epoch + 1):
#         file_path = os.path.join(output_dir, f'validation_data_epoch_{epoch}.pkl')
        
#         # Load the .pkl data
#         data = load_pkl_file(file_path)
        
#         # Extract and average per_sample_losses for each sample at this epoch
#         per_sample_losses = data['per_sample_losses']  # List of 10 tensors of shape (256, 24)
#         losses = [np.mean(tensor, axis=1) for tensor in per_sample_losses]  # Shape becomes (256, 1) for each batch
#         avg_losses = np.concatenate(losses)  # Shape becomes (2560, 1)
        
#         all_losses.append(avg_losses)

#     # Stack all the losses across epochs to get shape (num_epochs, 2560)
#     all_losses = np.stack(all_losses)  # Shape (num_epochs, 2560)

#     # Step 1: Calculate the mean loss for each datapoint across all epochs
#     mean_losses = np.mean(all_losses, axis=0)  # Shape (2560,)

#     return all_losses, mean_losses

# # Find indices of datapoints that have the bottom N% of average loss
# def get_bottom_percent_indices(mean_losses, quantile):
#     # Calculate the threshold for the given quantile
#     threshold = np.percentile(mean_losses, quantile)
    
#     # Get the indices of datapoints with loss less than the threshold
#     bottom_percent_indices = np.where(mean_losses < threshold)[0]
    
#     return bottom_percent_indices.tolist()  # Return as a list

# # Calculate average loss at each epoch using only the selected datapoints
# def calculate_average_loss_per_epoch(all_losses, selected_indices):
#     avg_losses_per_epoch = []
    
#     # For each epoch, calculate the mean loss for the selected indices
#     for epoch_losses in all_losses:
#         selected_losses = epoch_losses[selected_indices]
#         avg_loss = np.mean(selected_losses)
#         avg_losses_per_epoch.append(avg_loss)
    
#     return avg_losses_per_epoch

# # Plot the average validation loss over epochs for multiple quantiles
# def plot_average_loss_for_quantiles(avg_losses_per_epoch_by_quantile, quantiles, output_dir):
#     plt.figure(figsize=(10, 6))

#     for i, avg_losses_per_epoch in enumerate(avg_losses_per_epoch_by_quantile):
#         # Normalize the average losses
#         min_loss = np.min(avg_losses_per_epoch)
#         max_loss = np.max(avg_losses_per_epoch)
#         normalized_losses = (avg_losses_per_epoch - min_loss) / (max_loss - min_loss)
#         # normalized_losses = avg_losses_per_epoch
        
#         # Apply Savitzky-Golay filter for smoothing the curve
#         window_size = 91  # Choose an odd window size, adjust as needed
#         poly_order = 3    # Polynomial order, adjust as needed
#         smoothed_losses = savgol_filter(normalized_losses, window_size, poly_order)
#         # smoothed_losses = normalized_losses
        
#         # Plot the smoothed average validation loss with log scaling on the y-axis
#         # plt.plot(range(len(smoothed_losses)), smoothed_losses, marker='o', linestyle='-', label=f'{quantiles[i]}% quantile')
#         plt.plot(range(len(smoothed_losses)), smoothed_losses, linestyle='-', label=f'{quantiles[i]}% quantile', linewidth=1.0)
    
#     plt.yscale('log')  # Apply log scaling to the y-axis
#     plt.title("Average Validation Loss Over Epochs (Log Scale)")
#     plt.xlabel("Epochs")
#     plt.ylabel("Average Validation Loss (Log Scale)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
    
#     # Save the plot
#     plt.savefig(os.path.join('/iris/u/jubayer/diffusion_policy', 'smoothed_average_loss_multiple_quantiles_log_scaled.png'))
#     plt.show()

# # Main function to process all epochs, calculate mean losses, and plot average loss for selected datapoints
# def main(output_dir, start_epoch=0, end_epoch=3160):
#     # Step 1: Process the losses across all epochs and calculate mean losses per datapoint
#     all_losses, mean_losses = process_losses_across_epochs(output_dir, start_epoch, end_epoch)
    
#     # Quantiles you want to plot (e.g., 10%, 20%, 40%, 60%)
#     quantiles = [10, 20, 40, 60]
    
#     # Store average losses for each quantile
#     avg_losses_per_epoch_by_quantile = []
    
#     # Step 2: For each quantile, calculate the average loss for each epoch
#     for quantile in quantiles:
#         # bottom_percent_indices = get_bottom_percent_indices(mean_losses, quantile)
#         # print(f"For quantile = {quantile}, the indices are {bottom_percent_indices}"

#         avg_losses_per_epoch = calculate_average_loss_per_epoch(all_losses, bottom_percent_indices)
#         avg_losses_per_epoch_by_quantile.append(avg_losses_per_epoch)
    
#     # Step 3: Plot the average validation loss over epochs for all quantiles
#     plot_average_loss_for_quantiles(avg_losses_per_epoch_by_quantile, quantiles, output_dir)

# # Example usage
# output_dir = "/iris/u/jubayer/diffusion_policy/data/outputs/2024.09.12/07.32.23_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"  # Example directory
# main(output_dir, start_epoch=0, end_epoch=3300)

# output_dir = "/iris/u/jubayer/diffusion_policy/data/outputs/2024.09.11/10.04.41_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"  # Set your actual output directory path
# main(output_dir, start_epoch=0, end_epoch=286)


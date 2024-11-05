# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.mixture import GaussianMixture


# # Load data from a .pkl file
# def load_pkl_file(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     return data

# def calculate_loss(data):
    
#     val_data = data['val_data']
#     predicted_actions = data['val_data_pred']

#     # store all states and actions
#     all_states = []
#     all_actions = []
#     all_predicted_actions = []

#     # total_number_of_states = 13482 # for Push-T
#     total_number_of_states = 2405 # for Block Pushing

#     for index in range(total_number_of_states):
#     # for index in range(2405): # TODO: change this based on the total number of datapoints available -- (number of batches - 1) x (batch_size + 1 x number of elements in last batch)
#         # first find batch number 
#         batch_number = index // 256
#         # then find the index in the batch
#         index_in_batch = index % 256
#         # get the state
#         try:
#             state = val_data[batch_number]['obs'][index_in_batch]
#             all_states.append(state[0])
#             action = val_data[batch_number]['action'][index_in_batch]
#             all_actions.append(action[0])
#             predicted_action = predicted_actions[batch_number][index_in_batch]
#             all_predicted_actions.append(predicted_action[0])
#         except:
#             print(f"Indexing error: {index}")
    
#     # import ipdb
#     # ipdb.set_trace()

#     # Calculate loss
#     losses = []

#     for i in range(len(all_states)):
#         state = all_states[i]        
#         predicted_action = all_predicted_actions[i]

#         # set min_loss to positive infinity 
#         min_loss = float('inf')
        
#         for j in range(len(all_states)):
#             state_distance = np.linalg.norm(all_states[j] - state)
#             action_distance = np.linalg.norm(all_actions[j] - predicted_action)
            
#             # modified loss (1)
#             # candidate_loss = action_distance * np.exp(state_distance)

#             # modified loss (2)
#             candidate_loss = np.exp(action_distance) * np.exp(state_distance)
#             min_loss = min(min_loss, candidate_loss)
        
#         losses.append(min_loss)
    
#     # Calculate average loss
#     average_loss = np.mean(losses)

#     return average_loss
        


# # Main function to process all epochs and generate heatmap
# def main(output_dir, start_epoch, end_epoch):
#     all_indices = []
#     all_losses = []

#     # Process each epoch
#     for epoch in range(start_epoch, end_epoch + 1):
#         file_path = os.path.join(output_dir, f'validation_data_epoch_{epoch}.pkl')
        
#         # Load and process the epoch's losses
#         data = load_pkl_file(file_path)
        
#         # ordered_by_loss = process_losses(data, zerotofive, fivetoten, tentofifteen, fifteentotwenty, twentytotwentifive, twentyfivetothirty, thirtytothirtyfive, thirtyfivetoforty, fortytofortyfive, fortyfivetofifty, fiftytofiftyfive, fiftyfivetosixty, sixtytosixtyfive, sixtyfivetoseventy, seventytoseventyfive, seventyfivetoeighty, eightytoeightyfive, eightyfivetoninety, ninetytoninetyfive, ninetyfivetohundred)
#         # print(ordered_by_loss)
        
#         val_loss = calculate_loss(data)
#         all_losses.append(val_loss)
    
#     # plot all_losses against epoch
#     plt.plot(all_losses)
#     plt.xlabel('Epoch')
#     plt.ylabel('Validation Loss')
#     plt.title('Validation Loss vs Epoch')

#     # save the plot as modified_val_loss.png in the directory /iris/u/jubayer/diffusion_policy 
#     plt.savefig('/iris/u/jubayer/diffusion_policy/modified_val_loss.png')


# # Example usage

# output_dir = "/iris/u/jubayer/diffusion_policy/data/outputs/2024.10.30/00.53.01_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"

# # output_dir = "/iris/u/jubayer/diffusion_policy/data/outputs/2024.09.17/03.22.10_train_diffusion_unet_lowdim_pusht_lowdim" 
# main(output_dir, start_epoch=0, end_epoch=1451)


import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Load data from a .pkl file
def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


# Calculate loss using GPU acceleration
def calculate_loss_gpu(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_data = data['val_data']
    predicted_actions = data['val_data_pred']

    # Store all states and actions in PyTorch tensors
    all_states = []
    all_actions = []
    all_predicted_actions = []

    total_number_of_states = 2405  # for Block Pushing

    for index in range(total_number_of_states):
        batch_number = index // 256
        index_in_batch = index % 256
        try:
            state = val_data[batch_number]['obs'][index_in_batch]
            all_states.append(torch.tensor(state[0], device=device))
            action = val_data[batch_number]['action'][index_in_batch]
            all_actions.append(torch.tensor(action[0], device=device))
            predicted_action = predicted_actions[batch_number][index_in_batch]
            all_predicted_actions.append(torch.tensor(predicted_action[0], device=device))
        except:
            print(f"Indexing error: {index}")

    all_states = torch.stack(all_states)
    all_actions = torch.stack(all_actions)
    all_predicted_actions = torch.stack(all_predicted_actions)

    # Compute losses
    final_losses = []
    state_distances_list = []
    for i in range(len(all_states)):
        s = all_states[i]
        a_pred = all_predicted_actions[i]

        # Compute the loss for the current (s, a_pred) with all other (s', a')
        state_distances = torch.norm(all_states - s, dim=1, p=2) ** 2  # ||s' - s||^2
        action_distances = torch.norm(all_actions - a_pred, dim=1, p=2) ** 2  # ||a' - a_pred||^2

        # Compute the modified loss for all (s', a')
        k = 10
        modified_losses = k * action_distances * torch.exp(state_distances)

        # Find the minimum modified loss for the current (s, a_pred)
        min_loss = torch.min(modified_losses)

        # find the index of the minimum loss
        min_loss_index = torch.argmin(modified_losses)
        # find the state distance and action distance corresponding to the minimum loss
        state_distance = state_distances[min_loss_index]

        final_losses.append(min_loss)
        state_distances_list.append(state_distance)

    # Calculate average loss
    average_loss = torch.mean(torch.stack(final_losses)).item()

    # find maximum state distance
    max_state_distance = torch.mean(torch.stack(state_distances_list)).item()
    return average_loss, max_state_distance


# Main function to process all epochs and generate heatmap
def main(output_dir, start_epoch, end_epoch):
    all_losses = []
    all_max_state_distances = []

    for epoch in range(start_epoch, end_epoch + 1):
        file_path = os.path.join(output_dir, f'validation_data_epoch_{epoch}.pkl')

        # Load and process the epoch's losses
        data = load_pkl_file(file_path)

        val_loss, max_state_distance = calculate_loss_gpu(data)
        all_losses.append(val_loss)
        all_max_state_distances.append(max_state_distance)

    # Plot all_losses against epoch by smoothing the curve
    # smooth the curve
    window_size = 50
    smooth_losses = np.convolve(all_losses, np.ones(window_size) / window_size, mode='valid')
    plt.plot(smooth_losses, label='Smoothed Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epoch')
    plt.legend()
    
    plt.savefig('/iris/u/jubayer/diffusion_policy/modified_val_loss.png')

    # close the plot
    plt.close()



    # make a new plot for max state distance
    plt.plot(all_max_state_distances)
    plt.xlabel('Epoch')
    plt.ylabel('Max State Distance')
    plt.title('Max State Distance vs Epoch')
    plt.savefig('/iris/u/jubayer/diffusion_policy/max_state_distance.png')



# Example usage
output_dir = "/iris/u/jubayer/diffusion_policy/data/outputs/2024.10.30/00.53.01_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs"
main(output_dir, start_epoch=0, end_epoch=1451)

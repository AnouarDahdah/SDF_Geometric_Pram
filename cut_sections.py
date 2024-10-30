

Open In Colab

# Check shapes of SDF values
print("Original SDF shape:", sdf_values.shape)
print("Refined SDF shape over time:", refined_sdf_values_over_time.shape)

# Reshape the original SDF values for visualization
original_sdf_reshaped = sdf_values.numpy().reshape((grid_resolution, grid_resolution, grid_resolution))

# Define the slice indices to visualize
vertical_slice_index = grid_resolution // 2  # Middle index for vertical cut
profile_slice_index = grid_resolution // 2    # Middle index for profile cut
horizontal_slice_index = grid_resolution // 2  # Middle index for horizontal cut

# Create the refined SDF visualization over 20 time steps
plt.figure(figsize=(20, 12))

# Vertical Cuts
plt.subplot(3, 1, 1)  # One row, one column, and this is the first plot
for i in range(20):
    # Select the refined SDF slice
    refined_sdf_slice = refined_sdf_values_over_time[i][:, 0].reshape((grid_resolution, grid_resolution, grid_resolution))

    # Plot a vertical slice (along the y-axis)
    plt.imshow(refined_sdf_slice[:, vertical_slice_index, :].T, cmap='viridis', origin='lower')  # Transposed vertical slice
    plt.title(f'Vertical Cut at Time = {i + 1}')  # Changed to display time as 1, 2, ..., 20
    plt.colorbar(label='SDF Value')
    plt.axis('off')
    plt.pause(0.1)  # Pause to allow visualization

plt.tight_layout()
plt.show()

# Profile Cuts
plt.figure(figsize=(20, 12))

# Profile Cuts
plt.subplot(3, 1, 1)  # One row, one column, and this is the first plot
for i in range(20):
    # Select the refined SDF slice
    refined_sdf_slice = refined_sdf_values_over_time[i][:, 0].reshape((grid_resolution, grid_resolution, grid_resolution))

    # Plot a profile slice (along the x-axis)
    plt.imshow(refined_sdf_slice[profile_slice_index, :, :].T, cmap='viridis', origin='lower')  # Transposed profile slice
    plt.title(f'Profile Cut at Time = {i + 1}')  # Changed to display time as 1, 2, ..., 20
    plt.colorbar(label='SDF Value')
    plt.axis('off')
    plt.pause(0.1)  # Pause to allow visualization

plt.tight_layout()
plt.show()

# Horizontal Cuts
plt.figure(figsize=(20, 12))

# Horizontal Cuts
plt.subplot(3, 1, 1)  # One row, one column, and this is the first plot
for i in range(20):
    # Select the refined SDF slice
    refined_sdf_slice = refined_sdf_values_over_time[i][:, 0].reshape((grid_resolution, grid_resolution, grid_resolution))

    # Plot a horizontal slice (along the z-axis)
    plt.imshow(refined_sdf_slice[:, :, horizontal_slice_index].T, cmap='viridis', origin='lower')  # Transposed horizontal slice
    plt.title(f'Horizontal Cut at Time = {i + 1}')  # Changed to display time as 1, 2, ..., 20
    plt.colorbar(label='SDF Value')
    plt.axis('off')
    plt.pause(0.1)  # Pause to allow visualization

plt.tight_layout()
plt.show()

     

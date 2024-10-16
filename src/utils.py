import matplotlib.pyplot as plt  # Importing the plotting library to create and save plots.
import os  # Used for handling file paths and saving the plot to a specified directory.

def plot_loss(train_losses, val_losses, epoch, model_name, path_save_plot):
    # Create a figure with specific size (13x5 inches).
    fig = plt.figure(figsize=(13, 5))
    
    # Get the current axis from the figure.
    ax = fig.gca()
    
    # Enable interactive mode so that the plot can be updated in real time (optional if you are plotting during training).
    plt.ion()
    
    # Plot the training losses with a blue line.
    ax.plot(train_losses, label="Train loss", color="tab:blue")
    
    # Plot the validation losses with an orange line.
    ax.plot(val_losses, label="Validation loss", color="tab:orange")
    
    # Add a legend to differentiate between training and validation losses, with a specific font size.
    ax.legend(fontsize="16")
    
    # Set the x-axis label to "Epochs" with a font size of 16.
    ax.set_xlabel("Epochs", fontsize="16")
    
    # Set the y-axis label to "Loss" with a font size of 16.
    ax.set_ylabel("Loss", fontsize="16")
    
    # Set the title of the plot, including the model name, with a font size of 16.
    ax.set_title(f"Training and Validation Loss for {model_name}", fontsize="16")
    
    # Customize the font size for x and y ticks.
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Ensure the legend has a consistent font size for both training and validation losses.
    plt.legend(fontsize=12)
    
    # Add horizontal grid lines to the plot, with a dashed style and some transparency (alpha).
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Access the current axis again to modify its appearance.
    ax = plt.gca()
    
    # Hide the top and right spines (the border around the plot area).
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Set the thickness of the bottom and left spines to 0.5.
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    
    # Set the tick (the small lines along the axes) width to 0.5 for both axes.
    ax.tick_params(width=0.5)
    
    # Set the background color of the plot to a light grey ("whitesmoke").
    ax.set_facecolor("whitesmoke")
    
    # Create the filename for the plot by appending ".png" to the model name.
    model = model_name + ".png"
    
    # Create the full path where the plot will be saved.
    save_path = os.path.join(path_save_plot, model)
    
    # Save the plot to the specified path with a resolution of 300 dpi.
    plt.savefig(save_path, dpi=300)
    
    # Close the plot to free up memory after saving (especially important if plotting in loops).
    plt.close()

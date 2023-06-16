import matplotlib.pyplot as plt
import numpy as np


def save_plot_metrics(scored_data, zinc_data, attribute_col_name, saving_path):

    # Clean previous plots
    plt.clf()
    
    # Convert the attribute column to float
    zinc_data[attribute_col_name] = zinc_data[attribute_col_name].astype(float)
    scored_data[attribute_col_name] = scored_data[attribute_col_name].astype(float)

    # Compute normalized histograms
    zinc_hist, zinc_bins = np.histogram(zinc_data[attribute_col_name], bins=50, density=True)
    scored_hist, scored_bins = np.histogram(scored_data[attribute_col_name], bins=50, density=True)

    # Compute bin centers
    zinc_bin_centers = (zinc_bins[:-1] + zinc_bins[1:]) / 2
    gen_bin_centers = (scored_bins[:-1] + scored_bins[1:]) / 2

    # Plot histograms
    plt.plot(zinc_bin_centers, zinc_hist, label='ZINC', color='blue')
    plt.plot(gen_bin_centers, scored_hist, label='Generated', color='orange')

    # Add a legend to the plot
    plt.legend() 

    # Label the axes
    plt.xlabel(attribute_col_name)
    plt.ylabel('Frequency')
    
    # Save the figure

    plt.savefig(saving_path)
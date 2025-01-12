from scipy.stats import pearsonr, spearmanr
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class RegressionPlotter:
    '''
    A class to generate regression plots for each category.

    Methods:
    --------
    plot_predictions: Plots actual vs predicted values for a single category.
    plot_all_categories: Plots actual vs predicted values for all categories.
    '''
    def __init__(self):
        pass

    @staticmethod
    def plot_predictions(actual: np.array, predicted: np.array, category_index: int):
        '''
        Plot actual vs. predicted values for a given category.

        Args:
        -----
        actual (np.array): Array of actual values.
        predicted (np.array): Array of predicted values.
        category_index (int): The index of the category being plotted.
        '''
        # Calculate Pearson correlation and p-value
        corr, p_value = pearsonr(actual, predicted)

        # Create scatter plot
        plt.style.use('dark_background')
        plt.figure(figsize=(6, 6))
        plt.scatter(actual, predicted, color='cornflowerblue', alpha=0.8)

        # Plot y=x reference line
        x = np.linspace(min(actual), max(actual), 100)
        plt.plot(x, x, color='purple', linestyle='-', linewidth=2)

        # Add correlation coefficient and p-value as text
        plt.text(min(actual) + 1, max(predicted) - 1, f'corr= {corr:.2f}', 
                 fontsize=12, color='cyan', bbox=dict(facecolor='black', alpha=0.5))
        plt.text(min(actual) + 1, max(predicted) - 2, f'p= {p_value:.2e}', 
                 fontsize=12, color='yellow', bbox=dict(facecolor='black', alpha=0.5))

        # Set labels and title
        plt.xlabel('Actual', fontsize=14)
        plt.ylabel('Predicted', fontsize=14)
        plt.title(f'Category {category_index}: Actual vs Predicted', fontsize=16)

        # Adjust plot limits
        plt.xlim(min(actual) - 1, max(actual) + 1)
        plt.ylim(min(predicted) - 1, max(predicted) + 1)

        plt.show()

    def plot_all_categories(self, labels: np.array, probs: np.array):
        '''
        Plot actual vs. predicted values for all categories.

        Args:
        -----
        labels (np.array): Array of ground truth labels.
        probs (np.array): Array of predicted probabilities.
        '''
        for i in range(labels.shape[1]):
            print(f"Plotting for Category {i}")
            self.plot_predictions(labels[:, i], probs[:, i], category_index=i)
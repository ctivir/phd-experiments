from matplotlib import pyplot as plt
import pandas as pd


class ResultAnalyzer:
    def __init__(self, output_file):
        """
        Initializes the ResultAnalyzer class.
        
        :param output_file: Path to the output CSV file containing the results.
        """
        self.df = pd.read_csv(output_file)
    
    def analyze_and_graph_results(self):
        """
        Analyzes the experiment results and generates graphs.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='response')  # Change 'response' to the column you want to analyze
        plt.title('Distribution of Predicted Emotional States')
        plt.show()
        
# Analyze the results (for experiment 1 in this case)
result_analyzer = ResultAnalyzer('output_experiment_1.csv')
result_analyzer.analyze_and_graph_results()

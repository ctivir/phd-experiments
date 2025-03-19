import pandas as pd
from scipy import stats

def t_test_expertise_reversal(df, alpha=0.05):
    """
    Perform t-tests and Pearson correlations for expertise reversal effect.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'expertise_reversal', 'ere_post_test', and 'pretest_score'.
        alpha (float): Significance level for hypothesis testing. Default is 0.05.
    
    Returns:
        pd.DataFrame: Summary of t-test results and Pearson correlations.
    """
    
    # Grouping data based on expertise level and learning method
    groups = {
        "high_expert_practice": df[df['expertise_reversal'] == 'high-expertise/practice'],
        "high_expert_example": df[df['expertise_reversal'] == 'high-expertise/worked example'],
        "low_expert_practice": df[df['expertise_reversal'] == 'low-expertise/practice'],
        "low_expert_example": df[df['expertise_reversal'] == 'low-expertise/worked example']
    }
    
    # Compute delta (post-test score - pre-test score)
    deltas = {key: group['ere_post_test'] - group['pretest_score'] for key, group in groups.items()}
    
    # Initialize result storage
    results = []
    
    # Function to perform T-test and store results
    def perform_t_test(group1, group2, label):
        t_stat, p_value = stats.ttest_ind(deltas[group1], deltas[group2], equal_var=False, nan_policy='omit')
        reject_null = p_value < alpha
        results.append({
            "Comparison": label,
            "T-statistic": round(t_stat, 2),
            "P-value": round(p_value, 4),
            "Significant?": "Yes" if reject_null else "No",
            "Mean (Group 1)": round(deltas[group1].mean(), 2),
            "Mean (Group 2)": round(deltas[group2].mean(), 2)
        })

    # Perform T-tests
    perform_t_test("low_expert_example", "low_expert_practice", "Low-expertise: Worked-example vs Practice")
    perform_t_test("high_expert_example", "high_expert_practice", "High-expertise: Worked-example vs Practice")

    # Pearson Correlation Analysis
    pearson_results = []
    for label, group in groups.items():
        if len(group) > 1:  # Pearson requires at least 2 data points
            corr, p = stats.pearsonr(group['pretest_score'], group['ere_post_test'])
            pearson_results.append({
                "Group": label.replace("_", " ").title(),
                "Pearson Coefficient": round(corr, 3),
                "P-value": round(p, 4),
                "Significant?": "Yes" if p < alpha else "No"
            })

    # Convert results to DataFrames
    t_test_df = pd.DataFrame(results)
    pearson_df = pd.DataFrame(pearson_results)

    # Display summary
    print("\n<<<<< T-Test Results >>>>>")
    print(t_test_df.to_string(index=False))

    print("\n<<<<< Pearson Correlation Results >>>>>")
    print(pearson_df.to_string(index=False))

    return t_test_df, pearson_df  # Returning DataFrames for further use

# llm experiments results
gemma_result = pd.read_csv("/Users/ctivir/projects/ml/ed_simulator/unchk/gemma2_9b_it_output.csv", delimiter=',')
llama_result = pd.read_csv("/Users/ctivir/projects/ml/ed_simulator/unchk/llama3_8b_8192_40_output.csv", delimiter=',')

df_results, df_correlations = t_test_expertise_reversal(llama_result)
df_results.to_csv("../data/results/t_test_results.csv", index=False)
df_correlations.to_csv("../data/results/pearson_correlations.csv", index=False)
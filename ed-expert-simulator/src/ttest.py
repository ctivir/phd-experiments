import pandas as pd
import numpy as np
from scipy import stats

alpha = 0.05

def tttest(personas):
    
    # expert reversal effect 
    practice_high_expert = personas[personas['expertise_reversal'].isin(['high-expertise/practice'])]
    worked_example_high_expert = personas[personas['expertise_reversal'].isin(['high-expertise/worked example'])]
    practice_low_expert = personas[personas['expertise_reversal'].isin(['low-expertise/practice'])]
    worked_example_low_expert= personas[personas['expertise_reversal'].isin(['low-expertise/worked example'])]

    # Delta = posttrst - pretest 
    delta_we_low = worked_example_low_expert['ere_post_test'] - worked_example_low_expert['pretest_score']
    delta_practice_low = practice_low_expert['ere_post_test'] - practice_low_expert['pretest_score']
    delta_we_high = worked_example_high_expert['ere_post_test'] - worked_example_high_expert['pretest_score']
    delta_practice_high = practice_high_expert['ere_post_test'] - practice_high_expert['pretest_score']

    
    # Perform the two-sample t-test
    # 1. Low-expertise Learners: Worked example VS practice
    t_stat, p_value = stats.ttest_ind(delta_we_low, delta_practice_low, equal_var=False)
    print(f"\nDelta mean low worked-example: {delta_we_low.mean()}")
    print(f"Delta mean low practice: {delta_practice_low.mean()}")
    
    # Output results
    print("\n<<<<< 1. Low-expertise Learners: worked-example VS practice >>>>>")
    print(f"T-statistic = {round(t_stat,2)}")
    print(f"P-value = {p_value}")

    # Interpret results
    if p_value < alpha:
        print("Reject the null hypothesis: The means are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The means are not significantly different.")

    # 2. Low-expertise Learners: Worked example VS practice
    t_stat, p_value = stats.ttest_ind(delta_we_high, delta_practice_high, equal_var=False)
    print(f"\nDelta mean high worked-example: {delta_we_high.mean()}")
    print(f"Delta mean high practice: {delta_practice_high.mean()}")

    # Output results
    print("\n<<<<< 2. High-expertise Learners: worked example VS practice >>>>>")
    print(f"T-statistic = {round(t_stat,2)}")
    print(f"P-value = {round(p_value,2)}")

    # Interpret results
    if p_value < alpha:
        print("Reject the null hypothesis: The means are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The means are not significantly different.")

    # Compute Pearson's correlation coefficient
    print("\nPearson's correlation")
    print("\n<<<<< Overall: Pre-Test VS Post-Test >>>>>")
    print(stats.pearsonr(personas['pretest_score'], personas['ere_post_test']))

    print("\n<<<<< Low expert learners - Practice: Pre-Test VS Post-Test >>>>>")
    print(stats.pearsonr(practice_low_expert['pretest_score'], practice_low_expert['ere_post_test']))

    print("\n<<<<< Low expert learners - Worked-Example: Pre-Test VS Post-Test >>>>>")
    print(stats.pearsonr(worked_example_low_expert['pretest_score'], worked_example_low_expert['ere_post_test']))

    print("\n<<<<< High expert learners - Worked-Example: Pre-Test VS HE Post-Test >>>>>")
    print(stats.pearsonr(worked_example_high_expert['pretest_score'], worked_example_high_expert['ere_post_test']))

    print("\n<<<<< High expert learners - Practice: Pre-Test VS HE Post-Test >>>>>")
    print(stats.pearsonr(practice_high_expert['pretest_score'], practice_high_expert['ere_post_test']))
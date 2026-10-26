"""
Excel tables for results

"""

import pandas as pd

def coefficient_table(result):
    """
    Create a coefficient table for each fitted model
    """
    ci = result.conf_int()

    return pd.DataFrame({
        "Coefficient": result.params,
        "Standard error": result.std_err,
        "t-statistic": result.tvalues,
        "p-value": result.pvalues,
        "95% CI lower": ci.iloc[:, 0],
        "95% CI upper": ci.iloc[:, 1]
        }).rename_axis("Parameter")



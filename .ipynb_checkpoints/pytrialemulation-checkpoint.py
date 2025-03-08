import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import glm
from typing import Optional, Dict, Any, List
from statsmodels.stats.sandwich_covariance import cov_cluster
from statsmodels.genmod.generalized_linear_model import PerfectSeparationWarning
import warnings
import math
import re
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

# Ignore PerfectSeparationWarning
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)


class TrialSequence:
    def __init__(self, estimand):
        self.estimand = estimand
        self.data = None
        self.Id = None
        self.period = None
        self.treatment = None
        self.outcome = None
        self.eligible = None
        self.switch_weight_model = None
        self.censor_weight_model = None
        self.weight_calculated_switch = False
        self.weight_calculated_censor = False

    def set_data(self, data, Id, period, treatment, outcome, eligible):
        """
        Equivalent to the R method set_data(...), which eventually calls data_manipulation(...).
        """
        self.Id = Id
        self.period = period
        self.treatment = treatment
        self.outcome = outcome
        self.eligible = eligible

        # Copy and rename columns.
        df = data.copy()
        df = df.rename(columns={
            Id: "id",
            period: "period",
            treatment: "treatment",
            outcome: "outcome",
            eligible: "eligible"
        }, inplace=False)

        # Create 'first' column to mark the first row per subject.
        df = df.sort_values(["id", "period"], ascending=True)
        df["first"] = (df["id"] != df["id"].shift(1)).astype(int)
        
        # Call data_manipulation. For ITT, use_censor should be False.
        use_censor = (self.estimand == "PP")
        df = data_manipulation(df, use_censor=use_censor)

        # Optionally, select only the original columns plus added ones
        # (Here we include all columns; adjust as needed.)
        self.data = df.copy()
        return self

    def set_switch_weight_model(self, numerator, denominator, model_fitter):
        """
        Sets the switch weight model.
        Constructs formula strings like "treatment ~ age" from the provided RHS.
        """
        if not isinstance(model_fitter, StatsGLMLogit):
            raise ValueError("model_fitter must be an instance of StatsGLMLogit")

        # Build full formula strings; note: self.treatment holds the original treatment variable name.
        numerator_formula = f"{self.treatment} {numerator}"
        denominator_formula = f"{self.treatment} {denominator}"
        self.switch_weight_model = {
            "numerator": numerator_formula,
            "denominator": denominator_formula,
            "fitter": model_fitter,
            "fitted": {}  # Will store fitted results here.
        }
        return self

    def set_censor_weight_model(self, censor_event, numerator, denominator, model_fitter, pool_models="numerator"):
        """
        Sets the censor weight model.
        Constructs formulas for censoring as: "1 - <censor_event> ~ ..." for numerator and denominator.
        """
        if not isinstance(model_fitter, StatsGLMLogit):
            raise ValueError("model_fitter must be an instance of StatsGLMLogit")

        # Use Patsy's I() operator to force evaluation of the transformation.
        numerator_formula = f"I(1 - {censor_event}) {numerator}"
        denominator_formula = f"I(1 - {censor_event}) {denominator}"
        
        self.censor_weight_model = {
            "numerator": numerator_formula,
            "denominator": denominator_formula,
            "fitter": model_fitter,
            "pool_models": pool_models,
            "censor_event": censor_event,
            "fitted": {}
        }
        return self

    def calculate_weights(self):
        """
        Overall wrapper to calculate switch and censor weights.
        Initializes overall weight 'wt' and multiplies by wtS and wtC as available.
        """
        if self.data is None:
            raise ValueError("No data has been set. Call set_data(...) first.")

        self.data["wt"] = 1.0

        if self.switch_weight_model is not None:
            self._calculate_switch_weights()
            if "wtS" in self.data.columns:
                self.data["wt"] *= self.data["wtS"]
            self.weight_calculated_switch = True

        if self.censor_weight_model is not None:
            self._calculate_censor_weights()
            if "wtC" in self.data.columns:
                self.data["wt"] *= self.data["wtC"]
            self.weight_calculated_censor = True

        return self

    def _calculate_switch_weights(self):
        """
        Fits logistic regression models for switch weights on subsets defined by am_1.
        Computes wtS = (1 - p_n)/(1 - p_d) for treatment == 0 and p_n/p_d for treatment == 1.
        """
        df = self.data
        sw = self.switch_weight_model
        if "fitted" not in sw:
            sw["fitted"] = {}
    
        # Create 'am_1' if not present: previous treatment (shifted by 1, with missing as 0)
        if "am_1" not in df.columns:
            df["am_1"] = df.groupby("id")["treatment"].shift(1).fillna(0)
    
        # For rows where previous treatment was 1
        idx_1 = df.index[df["am_1"] == 1]
        fit_n1 = sw["fitter"].fit_weights_model(
            data=df.loc[idx_1],
            formula=sw["numerator"],
            label="n1"
        )
        df.loc[idx_1, "p_n"] = fit_n1["fitted"]
        sw["fitted"]["n1"] = fit_n1
    
        fit_d1 = sw["fitter"].fit_weights_model(
            data=df.loc[idx_1],
            formula=sw["denominator"],
            label="d1"
        )
        df.loc[idx_1, "p_d"] = fit_d1["fitted"]
        sw["fitted"]["d1"] = fit_d1
    
        # For rows where previous treatment was 0
        idx_0 = df.index[df["am_1"] == 0]
        fit_n0 = sw["fitter"].fit_weights_model(
            data=df.loc[idx_0],
            formula=sw["numerator"],
            label="n0"
        )
        df.loc[idx_0, "p_n"] = fit_n0["fitted"]
        sw["fitted"]["n0"] = fit_n0
    
        fit_d0 = sw["fitter"].fit_weights_model(
            data=df.loc[idx_0],
            formula=sw["denominator"],
            label="d0"
        )
        df.loc[idx_0, "p_d"] = fit_d0["fitted"]
        sw["fitted"]["d0"] = fit_d0
    
        # Instead of using inplace fillna, assign the result
        df["p_n"] = df["p_n"].fillna(1.0)
        df["p_d"] = df["p_d"].fillna(1.0)
    
        # Initialize wtS and compute based on treatment group
        df["wtS"] = np.nan
        df.loc[df["treatment"] == 0, "wtS"] = (1.0 - df["p_n"]) / (1.0 - df["p_d"])
        df.loc[df["treatment"] == 1, "wtS"] = df["p_n"] / df["p_d"]
        df["wtS"] = df["wtS"].fillna(1.0)
    
        return self

    def _calculate_censor_weights(self):
        """
        Fits logistic regression models for censor weights.
        Uses pooled numerator but splits denominator by prior treatment if pool_models="numerator".
        Then computes wtC = pC_n / pC_d.
        """
        df = self.data
        cw = self.censor_weight_model
        if "fitted" not in cw:
            cw["fitted"] = {}

        censor_event = cw.get("censor_event", "censored")
        pool_models = cw.get("pool_models", "numerator")
        
        if pool_models != "both":
            if "am_1" not in df.columns:
                df["am_1"] = df.groupby("id")["treatment"].shift(1).fillna(0)
        
        df["pC_n"] = 1.0
        df["pC_d"] = 1.0

        if pool_models in ("numerator", "both"):
            fit_n = cw["fitter"].fit_weights_model(data=df,
                                                     formula=cw["numerator"],
                                                     label="n")
            df["pC_n"] = fit_n["fitted"]
            cw["fitted"]["n"] = fit_n
        else:
            idx_0 = df.index[df["am_1"] == 0]
            idx_1 = df.index[df["am_1"] == 1]
            fit_n0 = cw["fitter"].fit_weights_model(data=df.loc[idx_0],
                                                      formula=cw["numerator"],
                                                      label="n0")
            df.loc[idx_0, "pC_n"] = fit_n0["fitted"]
            cw["fitted"]["n0"] = fit_n0
            fit_n1 = cw["fitter"].fit_weights_model(data=df.loc[idx_1],
                                                      formula=cw["numerator"],
                                                      label="n1")
            df.loc[idx_1, "pC_n"] = fit_n1["fitted"]
            cw["fitted"]["n1"] = fit_n1

        if pool_models in ("denominator", "both"):
            fit_d = cw["fitter"].fit_weights_model(data=df,
                                                     formula=cw["denominator"],
                                                     label="d")
            df["pC_d"] = fit_d["fitted"]
            cw["fitted"]["d"] = fit_d
        else:
            idx_0 = df.index[df["am_1"] == 0]
            idx_1 = df.index[df["am_1"] == 1]
            fit_d0 = cw["fitter"].fit_weights_model(data=df.loc[idx_0],
                                                      formula=cw["denominator"],
                                                      label="d0")
            df.loc[idx_0, "pC_d"] = fit_d0["fitted"]
            cw["fitted"]["d0"] = fit_d0
            fit_d1 = cw["fitter"].fit_weights_model(data=df.loc[idx_1],
                                                      formula=cw["denominator"],
                                                      label="d1")
            df.loc[idx_1, "pC_d"] = fit_d1["fitted"]
            cw["fitted"]["d1"] = fit_d1
        
        df["pC_n"] = df["pC_n"].fillna(1.0)
        df["pC_d"] = df["pC_d"].fillna(1.0)
        df["wtC"] = df["pC_n"] / df["pC_d"]
        df["wtC"] = df["wtC"].fillna(1.0)

    def switch_weights_print(self):
        if not self.switch_weight_model:
            print("## Switch Weight Model: Not set")
            return
        print("##  - Numerator formula:", self.switch_weight_model["numerator"])
        print("##  - Denominator formula:", self.switch_weight_model["denominator"])
        print("##  - Model fitter type:", type(self.switch_weight_model["fitter"]).__name__)
        if not self.weight_calculated_switch:
            print("##  - Weight models not fitted. Use calculate_weights()")
        else:
            print("##  - Switch weights computed.")

    def censor_weights_print(self):
        if not self.censor_weight_model:
            print("## Censor Weight Model: Not set")
            return
        print("##  - Numerator formula:", self.censor_weight_model["numerator"])
        print("##  - Denominator formula:", self.censor_weight_model["denominator"])
        pool_models = self.censor_weight_model.get("pool_models", None)
        if pool_models == "numerator":
            print("##  - Numerator model is pooled across treatment arms. Denominator model is not pooled.")
        elif pool_models == "denominator":
            print("##  - Denominator model is pooled. Numerator model is not pooled.")
        elif pool_models == "both":
            print("##  - Both numerator and denominator models are pooled.")
        else:
            print("##  - Neither numerator nor denominator is pooled.")
        print("##  - Model fitter type:", type(self.censor_weight_model["fitter"]).__name__)
        if not self.weight_calculated_censor:
            print("##  - Weight models not fitted. Use calculate_weights()")
        else:
            print("##  - Censor weights computed.")

    def __repr__(self):
        if self.data is None:
            return f"Trial Sequence Object\nEstimand: {self.estimand}\nNo data loaded."
        num_obs = min(len(self.data), 10)
        num_patients = self.data["id"].nunique() if "id" in self.data.columns else "Unknown"
        data_preview = pd.concat([self.data.head(2), self.data.tail(2)])
        repr_str = f"## Trial Sequence Object\n"
        repr_str += f"## Estimand: {'Per-protocol' if self.estimand == 'PP' else 'Intention-to-treat'}\n\n"
        repr_str += f"## Data:\n"
        repr_str += f"  - Showing {2+2} of {len(self.data)} observations from {num_patients} patients\n"
        repr_str += f"{data_preview.to_string(index=False)}\n"
        if self.switch_weight_model is not None:
            repr_str += f"  - Numerator formula: {self.switch_weight_model['numerator']}\n"
            repr_str += f"  - Denominator formula: {self.switch_weight_model['denominator']}\n"
            repr_str += f"  - Model fitter type: {type(self.switch_weight_model['fitter']).__name__}\n"
            if not self.weight_calculated_switch:
                repr_str += f"  - Switch weight models not fitted. Use calculate_weights()\n"
            else:
                repr_str += f"  - Switch weights computed.\n"
        if self.censor_weight_model is not None:
            repr_str += f"  - Numerator formula: {self.censor_weight_model['numerator']}\n"
            repr_str += f"  - Denominator formula: {self.censor_weight_model['denominator']}\n"
            repr_str += f"  - Model fitter type: {type(self.censor_weight_model['fitter']).__name__}\n"
            if not self.weight_calculated_censor:
                repr_str += f"  - Censor weight models not fitted. Use calculate_weights()\n"
            else:
                repr_str += f"  - Censor weights computed.\n"
        return repr_str


def data_manipulation(df, use_censor=True):
    """
    Python equivalent to the R function data_manipulation(data, use_censor=TRUE).
    This function:
      1) Removes rows before the subject's first 'eligible=1' period.
      2) Removes rows after the subject's first outcome=1 period.
      3) Computes time_of_event, am_1 (lagged treatment), switch, regime_start, time_on_regime, cumA.
      4) If use_censor=True, calls censor_func(...) for additional switching-based censoring.
      5) Sets eligible0=1 if am_1==0, eligible1=1 if am_1==1.
    """
    df = df.copy()
    # Ensure the data is sorted and create 'first' column as in R.
    df.sort_values(["id", "period"], ascending=True, inplace=True)
    df["first"] = (df["id"] != df["id"].shift(1)).astype(int)

    # --- Remove rows before first eligible=1 ---
    def earliest_eligible_period(subdf):
        mask = (subdf["eligible"] == 1)
        if not mask.any():
            return np.inf
        return subdf.loc[mask, "period"].min()
    min_eligible_df = (df.groupby("id", as_index=False)
                        .apply(earliest_eligible_period)
                        .rename(columns={None: "min_eligible_period"}))
    df = df.merge(min_eligible_df, on="id", how="left")
    before_len = len(df)
    df = df[df["period"] >= df["min_eligible_period"]]
    after_len = len(df)
    if after_len < before_len:
        removed_count = before_len - after_len
        print(f"Warning: Removed {removed_count} rows before trial eligibility.")
    df.drop(columns=["min_eligible_period"], inplace=True)

    # --- Remove rows after first outcome=1 ---
    def earliest_outcome_period(subdf):
        mask = (subdf["outcome"] == 1)
        if not mask.any():
            return np.inf
        return subdf.loc[mask, "period"].min()
    min_outcome_df = (df.groupby("id", as_index=False)
                       .apply(earliest_outcome_period)
                       .rename(columns={None: "min_outcome_period"}))
    df = df.merge(min_outcome_df, on="id", how="left")
    before_len = len(df)
    df = df[df["period"] <= df["min_outcome_period"]]
    after_len = len(df)
    if after_len < before_len:
        removed_count = before_len - after_len
        print(f"Warning: Removed {removed_count} rows after outcome occurred.")
    df.drop(columns=["min_outcome_period"], inplace=True)

    # --- Compute time_of_event ---
    df["time_of_event"] = 9999
    outcome_mask = (df["outcome"] == 1)
    df.loc[outcome_mask, "time_of_event"] = df.loc[outcome_mask, "period"]

    # --- Compute am_1, switch, regime_start, time_on_regime, cumA ---
    df["am_1"] = np.nan
    df["switch"] = 0
    df["regime_start"] = np.nan
    df["time_on_regime"] = 0.0
    df["cumA"] = 0.0

    def process_id(sub):
        sub = sub.sort_values("period").copy()
        sub["am_1"] = sub["treatment"].shift(1).fillna(0)
        sub.iloc[0, sub.columns.get_loc("switch")] = 0
        sub.iloc[0, sub.columns.get_loc("regime_start")] = sub["period"].iloc[0]
        sub.iloc[0, sub.columns.get_loc("time_on_regime")] = 0.0
        sub.iloc[0, sub.columns.get_loc("cumA")] = sub["treatment"].iloc[0]
        for i in range(1, len(sub)):
            prev_am = sub["am_1"].iloc[i]
            curr_trt = sub["treatment"].iloc[i]
            if prev_am != curr_trt:
                sub.iloc[i, sub.columns.get_loc("switch")] = 1
                sub.iloc[i, sub.columns.get_loc("regime_start")] = sub["period"].iloc[i]
            else:
                sub.iloc[i, sub.columns.get_loc("switch")] = 0
                sub.iloc[i, sub.columns.get_loc("regime_start")] = sub["regime_start"].iloc[i - 1]
            sub.iloc[i, sub.columns.get_loc("time_on_regime")] = sub["period"].iloc[i] - sub["regime_start"].iloc[i]
            sub.iloc[i, sub.columns.get_loc("cumA")] = sub["treatment"].iloc[i]
        sub["cumA"] = sub["cumA"].cumsum()
        return sub

    df = df.groupby("id", group_keys=False).apply(process_id).reset_index(drop=True)

    # --- If use_censor is True, call censor_func ---
    if use_censor:
        df = censor_func(df)

    # --- Set eligible0 and eligible1 ---
    df["eligible0"] = 0
    df["eligible1"] = 0
    df.loc[df["am_1"] == 0, "eligible0"] = 1
    df.loc[df["am_1"] == 1, "eligible1"] = 1

    return df


def censor_func(df):
    """
    Python replication of the C++ censoring function.
    This function expects the DataFrame to have the following columns:
      ["first", "eligible", "treatment", "switch", "started0", "started1",
       "stop0", "stop1", "eligible0_sw", "eligible1_sw", "delete"]
    It updates the "delete" flag based on the logic in the C++ code.
    """
    required_cols = ["first", "eligible", "treatment", "switch",
                     "started0", "started1", "stop0", "stop1",
                     "eligible0_sw", "eligible1_sw", "delete"]
    for col in required_cols:
        if col not in df.columns:
            if col == "delete":
                df[col] = False
            else:
                df[col] = 0

    n = len(df)
    started0_ = 0
    started1_ = 0
    stop0_ = 0
    stop1_ = 0
    eligible0_sw_ = 0
    eligible1_sw_ = 0

    # Loop over rows to mimic C++ logic.
    for i in range(n):
        if df.at[i, "first"] == 1:
            started0_ = 0
            started1_ = 0
            stop0_ = 0
            stop1_ = 0
            eligible0_sw_ = 0
            eligible1_sw_ = 0
        if stop0_ == 1 or stop1_ == 1:
            started0_ = 0
            started1_ = 0
            stop0_ = 0
            stop1_ = 0
            eligible0_sw_ = 0
            eligible1_sw_ = 0
        if started0_ == 0 and started1_ == 0 and df.at[i, "eligible"] == 1:
            if df.at[i, "treatment"] == 0:
                started0_ = 1
            elif df.at[i, "treatment"] == 1:
                started1_ = 1
        if started0_ == 1 and stop0_ == 0:
            eligible0_sw_ = 1
            eligible1_sw_ = 0
        elif started1_ == 1 and stop1_ == 0:
            eligible0_sw_ = 0
            eligible1_sw_ = 1
        else:
            eligible0_sw_ = 0
            eligible1_sw_ = 0
        if df.at[i, "switch"] == 1:
            if df.at[i, "eligible"] == 1:
                if df.at[i, "treatment"] == 1:
                    started1_ = 1
                    stop1_ = 0
                    started0_ = 0
                    stop0_ = 0
                    eligible1_sw_ = 1
                elif df.at[i, "treatment"] == 0:
                    started0_ = 1
                    stop0_ = 0
                    started1_ = 0
                    stop1_ = 0
                    eligible0_sw_ = 1
            else:
                stop0_ = started0_
                stop1_ = started1_
        if eligible0_sw_ == 0 and eligible1_sw_ == 0:
            df.at[i, "delete"] = True
        else:
            df.at[i, "started0"] = started0_
            df.at[i, "started1"] = started1_
            df.at[i, "stop0"] = stop0_
            df.at[i, "stop1"] = stop1_
            df.at[i, "eligible0_sw"] = eligible0_sw_
            df.at[i, "eligible1_sw"] = eligible1_sw_
            df.at[i, "delete"] = False

    df = df.loc[df["delete"] == False].copy()
    return df


def add_rhs(f1: str, f2: str) -> str:
    """
    Combine two RHS formulas.

    Parameters:
      f1: A formula string (e.g., "~ a + b")
      f2: A formula string (e.g., "z ~ c + log(d)") or "~ c + log(d)"

    Returns:
      A combined formula string of the form "~ <rhs(f1)> + <rhs(f2)>"
    """
    # Remove the leading '~' and any surrounding whitespace.
    def clean_formula(f: str) -> str:
        f = f.strip()
        if f.startswith("~"):
            f = f[1:].strip()
        # If the formula contains a '~', split and take the RHS.
        if "~" in f:
            _, rhs = f.split("~", 1)
            return rhs.strip()
        return f

    rhs1 = clean_formula(f1)
    rhs2 = clean_formula(f2)
    # Combine the two RHS parts.
    combined = f"~ {rhs1} + {rhs2}"
    return combined


def as_formula(x) -> str:
    if isinstance(x, str):
        if x.strip().startswith("~"):
            return "~ " + x.strip().lstrip("~").strip()
        else:
            return "~ " + x.strip()
    elif isinstance(x, list):
        return "~ " + " + ".join(x)
    else:
        raise ValueError("Unsupported type for formula conversion.")


def get_stabilised_weights_terms(trial_seq) -> str:
    terms = "~1"
    if trial_seq.censor_weight_model is not None:
        covar = extract_vars(trial_seq.censor_weight_model["numerator"])
        if covar:
            terms = add_rhs(terms, "~ " + " + ".join(covar))
    if trial_seq.switch_weight_model is not None:
        covar = extract_vars(trial_seq.switch_weight_model["numerator"])
        if covar:
            terms = add_rhs(terms, "~ " + " + ".join(covar))
    return terms
    

def extract_vars(formula_str: str) -> List[str]:
    """
    Given a formula string, return the list of variable names appearing on the right-hand side.
    If the formula is simply "1" (or "~1"), return an empty list.
    """
    formula_str = formula_str.strip()
    if formula_str in ["1", "~1"]:
        return []
    if "~" in formula_str:
        _, rhs = formula_str.split("~", 1)
    else:
        rhs = formula_str
    # Remove any I() wrappers and parentheses.
    rhs = rhs.replace("I(", "").replace(")", "")
    tokens = re.findall(r"[A-Za-z_]\w*", rhs)
    # Remove any "1" if present.
    return [token for token in tokens if token != "1"]


def update_outcome_formula(trial_seq):
    """
    Combines the individual outcome model formula parts into a single well-formed formula.
    It removes any redundant intercept specifications and duplicate predictor names.
    """
    outcome_model = trial_seq.outcome_model  # assumed to be a dict

    parts = []
    # Collect parts for the right-hand side.
    for key in ["treatment_terms", "adjustment_terms", "followup_time_terms", "trial_period_terms", "stabilised_weights_terms"]:
        part = outcome_model.get(key, "")
        if part:
            # Remove the leading "~" and extra spaces.
            part_clean = part.lstrip("~").strip()
            # Skip if the part is empty or just "1".
            if part_clean and part_clean != "1":
                parts.append(part_clean)
    # Remove duplicates while preserving order.
    unique_parts = []
    seen = set()
    for p in parts:
        if p not in seen:
            unique_parts.append(p)
            seen.add(p)
    # Join the parts with " + " to form the RHS.
    rhs = " + ".join(unique_parts)
    # Build the final formula; note that patsy will automatically add an intercept.
    formula_str = f"outcome ~ {rhs}"
    outcome_model["formula"] = formula_str

    # For adjustment variables, extract from the adjustment_terms and stabilised_weights_terms.
    adj_vars = extract_vars(outcome_model.get("adjustment_terms", "")) + extract_vars(outcome_model.get("stabilised_weights_terms", ""))
    outcome_model["adjustment_vars"] = list(set(adj_vars))

    trial_seq.outcome_model = outcome_model
    return trial_seq


def set_outcome_model(trial_seq, treatment_var: Optional[str] = None,
                      adjustment_terms: str = "~1",
                      followup_time_terms: str = "~ followup_time + I(followup_time**2)",
                      trial_period_terms: str = "~ trial_period + I(trial_period**2)",
                      model_fitter: Optional[Any] = None) -> Any:
    if model_fitter is None:
        model_fitter = StatsGLMLogit()  # default model fitter

    # For ITT and PP, use "assigned_treatment" as the treatment variable.
    if trial_seq.estimand in ["ITT", "PP"]:
        treatment_term = "~ assigned_treatment"
        if "assigned_treatment" not in trial_seq.data.columns:
            trial_seq.data["assigned_treatment"] = trial_seq.data["treatment"]
    else:
        treatment_term = f"~ {treatment_var}" if treatment_var is not None else "~ dose"

    outcome_model = {
        "treatment_terms": treatment_term,
        "adjustment_terms": adjustment_terms,
        "followup_time_terms": followup_time_terms,
        "trial_period_terms": trial_period_terms,  # now uses trial_period_centered
        "stabilised_weights_terms": "~1",
        "model_fitter": model_fitter,
        "treatment_var": "assigned_treatment"
    }
    trial_seq.outcome_model = outcome_model
    trial_seq = update_outcome_formula(trial_seq)
    return trial_seq


class ExpansionOptions:
    def __init__(self, output, chunk_size, first_period=0, last_period=math.inf, censor_at_switch=False):
        self.chunk_size = chunk_size
        self.datastore = output
        self.first_period = first_period
        self.last_period = last_period
        self.censor_at_switch = censor_at_switch

    def __repr__(self):
        s = "Sequence of Trials Data:\n"
        s += f"  - Chunk size: {self.chunk_size}\n"
        s += f"  - Censor at switch: {self.censor_at_switch}\n"
        s += f"  - First period: {self.first_period} | Last period: {self.last_period}\n"
        s += f"  - Datastore contains {len(self.datastore)} chunk(s)"
        if self.datastore:
            chunk = self.datastore[0]
            s += "\n\nA TE Datastore Datatable object\n"
            s += f"N: {len(chunk)} observations\n"
            s += f"{pd.concat([chunk.head(2), chunk.tail(2)]).to_string(index=False)}"
        return s


def save_to_datatable():
    return []


def save_expanded_data(datastore, expanded_chunk):
    datastore.append(expanded_chunk)
    return datastore


def set_expansion_options(trial_seq, output, chunk_size, first_period=0, last_period=math.inf):
    trial_seq.expansion = ExpansionOptions(output=output,
                                            chunk_size=chunk_size,
                                            first_period=first_period,
                                            last_period=last_period,
                                            censor_at_switch=(trial_seq.estimand == "PP"))
    return trial_seq


def expand_until_switch(s, n=None):
    """
    Mimics R's expand_until_switch:
      first_switch <- match(1, s)
      if (!is.na(first_switch)) rep(c(1, 0), times = c(first_switch - 1, n - first_switch + 1))
      else rep(1, n)
    (Adjusted for Python’s 0-based indexing.)
    """
    s = np.array(s)
    if np.any(s == 1):
        first_switch = np.argmax(s == 1) + 1
    else:
        first_switch = len(s) + 1
    pre = np.ones(first_switch - 1, dtype=int)
    post = np.zeros(len(s) - (first_switch - 1), dtype=int)
    return np.concatenate((pre, post))

    
def expand(sw_data, outcomeCov_vars, where_var, use_censor, minperiod, maxperiod, keeplist):

    # --- Step 1: Compute Weights Correctly ---
    sw_data = sw_data.copy()
    # For the first row per subject, initialize weight0 = 1.0
    sw_data.loc[sw_data["first"] == 1, "weight0"] = 1.0
    # Compute cumulative product of weights within each ID
    sw_data["weight0"] = sw_data.groupby("id")["wt"].cumprod()
    # Compute wtprod: shift weight0 one period back (first row gets fill value 1.0)
    sw_data["wtprod"] = sw_data.groupby("id")["weight0"].shift(1, fill_value=1.0)
    # Normalize weight0 by dividing by the first weight0 for each id
    sw_data["weight0"] = sw_data.groupby("id")["weight0"].transform(lambda x: x / x.iloc[0])

    # --- Step 2: Build temp_data with eligibility filtering ---
    temp_data = sw_data.copy()
    temp_data['expand'] = ((temp_data['eligible'] == 1) &
                           (temp_data['treatment'].notna()) &
                           (temp_data['period'] >= minperiod) &
                           (temp_data['period'] <= maxperiod)
                          ).astype(int)
    temp_data['period'] = temp_data['period'].astype(int)


    # --- Step 3: Replicate each row to create trial periods ---
    replicated_list = []
    for _, row in sw_data.iterrows():
        for tp in range(int(row["period"]) + 1):
            new_row = row.copy()
            new_row["trial_period"] = tp
            replicated_list.append(new_row)
    replicated = pd.DataFrame(replicated_list)
    # Rename columns: the original 'period' in sw_data becomes 'period_new' in replicated
    replicated = replicated.rename(columns={"period": "period_new", "switch": "switch_new"})
    replicated["trial_period"] = replicated["trial_period"].astype(int)

    if not use_censor:
        replicated["switch_new"] = 0

    # --- Step 4: Merge temp_data with replicated data ---
    # Here, we want to match the replicated trial_period to the temp_data's period.
    # That is, we merge on temp_data.period == replicated.trial_period.
    merged = replicated.merge(
        temp_data, left_on=["id", "trial_period"], right_on=["id", "period"],
        how="left", suffixes=("", "_temp")
    )


    # --- Step 5: Compute followup_time ---
    merged["followup_time"] = merged["period_new"] - merged["trial_period"]
    merged.loc[merged["followup_time"] == 0, "switch_new"] = 0
    # Use the temp_data's expand flag to filter rows
    merged = merged[merged["expand"] == 1].copy()


    # --- Step 6: Apply Censoring, if Enabled ---
    if use_censor:
        def apply_expand(group):
            s = group["switch_new"].values
            flags = expand_until_switch(s, n=len(s))
            group = group.copy()
            group["expand"] = flags
            return group
        merged = merged.groupby(["id", "trial_period"], group_keys=False).apply(apply_expand)
        merged = merged.reset_index(drop=True)
        merged = merged[merged["expand"] == 1]

    # --- Step 7: Compute Final Weights ---
    # In this version, we want the final weight to be the computed weight0
    merged["weight"] = merged["weight0"]
    merged["weight"] = merged["weight"].fillna(1.0)
    # --- Step 8: Return Required Columns ---
    expanded_df = merged[[col for col in keeplist if col in merged.columns]]
    return expanded_df


def expand_trials(trial_seq):
    """
    Expand the full dataset without chunking.
    """
    data = trial_seq.data.copy().sort_values(["id", "period"], ascending=True)

    # Get global bounds based on eligibility
    if (data["eligible"] == 1).any():
        global_min = data.loc[data["eligible"] == 1, "period"].min()
        global_max = data.loc[data["eligible"] == 1, "period"].max()
    else:
        global_min = data["period"].min()
        global_max = data["period"].max()

    first_period = max(trial_seq.expansion.first_period, global_min)
    last_period = min(trial_seq.expansion.last_period, global_max)

    outcome_adj_vars = trial_seq.outcome_model.get("adjustment_vars", [])
    if "x2" not in outcome_adj_vars:
        outcome_adj_vars.append("x2")

    keeplist = list(set(
        ["id", "trial_period", "followup_time", "outcome", "weight", "treatment"]
        + outcome_adj_vars
        + [trial_seq.outcome_model.get("treatment_var", "assigned_treatment")]
    ))

    if "wt" not in data.columns:
        data["wt"] = 1.0

    expanded_df = expand(
        sw_data=data,
        outcomeCov_vars=outcome_adj_vars,
        where_var=None,
        use_censor=trial_seq.expansion.censor_at_switch,  
        minperiod=first_period,
        maxperiod=last_period,
        keeplist=keeplist
    )


    # Save the full expanded dataset as a single DataFrame
    trial_seq.expansion.datastore = [expanded_df]
    return trial_seq


def read_expanded_data(datastore, period=None, subset_condition=None):
    """
    Reads all expanded data from a datastore (assumed to be a list of DataFrames) 
    and optionally subsets it by trial period and/or an additional condition.

    Parameters:
      datastore : list of pandas.DataFrame
          The expanded data chunks.
      period : int or list of ints, optional
          Trial period(s) to include (if provided).
      subset_condition : str, optional
          A query string to subset the data (e.g., "age > 65 & followup_time <= 20").
    
    Returns:
      pandas.DataFrame with the expanded data.
    """
    # Combine all chunks into one DataFrame.
    data_table = pd.concat(datastore, ignore_index=True)
    
    # Filter by trial period if specified.
    if period is not None:
        if np.isscalar(period):
            period = [period]
        data_table = data_table[data_table['trial_period'].isin(period)]
    
    # If a subset condition is provided, apply it as a query.
    if subset_condition is not None:
        data_table = data_table.query(subset_condition)
    
    return data_table


def sample_expanded_data(datastore, p_control, period=None, subset_condition=None, seed=None):
    """
    Reads and samples the expanded data from the datastore.
    For observations with outcome == 0 (controls), each is kept with probability p_control.
    A seed is used for reproducibility.

    Parameters:
      datastore : list of pandas.DataFrame
          The expanded data chunks.
      p_control : float
          The probability of selecting a control (outcome == 0).
      period : int or list of ints, optional
          Trial period(s) to include.
      subset_condition : str, optional
          An optional query to subset the data.
      seed : int, optional
          A random seed for reproducibility.
    
    Returns:
      A pandas.DataFrame of the (sampled) expanded data with a 'sample_weight' column set to 1.
    """
    # First, read (and combine) all expanded data.
    data_table = read_expanded_data(datastore, period=period, subset_condition=subset_condition)
    
    # Set the seed for reproducibility.
    if seed is not None:
        np.random.seed(seed)
    
    # Identify control rows (assume outcome==0 means control).
    is_control = data_table['outcome'] == 0
    # Generate a random number for each row.
    rand_vals = np.random.rand(len(data_table))
    # For control rows, keep if random number < p_control; otherwise keep treatment rows.
    keep_mask = (~is_control) | (rand_vals < p_control)
    sampled = data_table[keep_mask].copy()
    sampled['sample_weight'] = 1  # set sample weight to 1 for all rows.
    
    return sampled


def load_expanded_data(trial_seq, p_control=None, period=None, subset_condition=None, seed=None):
    """
    Loads (or samples) the expanded trial data and stores it in trial_seq.outcome_data.
    
    If p_control is None, all expanded data are loaded; otherwise, controls (rows with outcome==0)
    are sampled with probability p_control.
    
    Parameters:
      trial_seq : an object with an attribute trial_seq.expansion.datastore (a list of DataFrames)
          and an attribute trial_seq.outcome_data.
      p_control : float or None, optional
          The probability of selecting a control (if None, no sampling is performed).
      period : int or list of ints, optional
          Trial period(s) to include.
      subset_condition : str, optional
          A query condition to subset the data.
      seed : int, optional
          Random seed for reproducibility.
    
    Returns:
      The updated trial_seq object with trial_seq.outcome_data set to the loaded (or sampled) DataFrame.
    """
    if not trial_seq.expansion.datastore:
        raise ValueError("No expanded data found in the datastore. Please run expand_trials() first.")
    
    if p_control is None:
        loaded = read_expanded_data(trial_seq.expansion.datastore, period=period, subset_condition=subset_condition)
        loaded['sample_weight'] = 1
    else:
        loaded = sample_expanded_data(trial_seq.expansion.datastore,
                                      p_control=p_control,
                                      period=period,
                                      subset_condition=subset_condition,
                                      seed=seed)
    
    trial_seq.outcome_data = loaded
    return trial_seq


def fit_msm(trial_seq, weight_cols=['weight', 'sample_weight'], modify_weights=None):
    # Get the expanded outcome data.
    data = trial_seq.outcome_data.copy()
    
    # Compute the overall weight as the product of the specified weight columns.
    if weight_cols:
        w = np.ones(len(data))
        for col in weight_cols:
            w *= data[col]
    else:
        w = np.ones(len(data))
    
    # Apply weight transformation if provided.
    if modify_weights is not None:
        w = modify_weights(w)
    
    # Retrieve the outcome model formula.
    formula = trial_seq.outcome_model.get("formula", None)
    if formula is None:
        raise ValueError("No outcome model formula found in trial_seq.outcome_model. Please set the outcome model first.")
    
    # Retrieve the model fitter.
    model_fitter = trial_seq.outcome_model.get("model_fitter", None)
    
    # Fit the model.
    if model_fitter is not None and hasattr(model_fitter, "fit_outcome_model"):
        fitted = model_fitter.fit_outcome_model(data, formula, weights=w, cluster_col="id")
    else:
        model = smf.glm(formula=formula, data=data, family=smf.families.Binomial(), freq_weights=w)
        fitted_raw = model.fit(cov_type='cluster', cov_kwds={'groups': data['id'], 'use_correction': True})
        from your_helper_module import tidy_statmodels_result, glance_statmodels_result  # adjust import as needed
        tidy_df = tidy_statmodels_result(fitted_raw)
        glance_dict = glance_statmodels_result(fitted_raw)
        fitted = {
            "model": {"params": fitted_raw.params, "cov_params": fitted_raw.cov_params()},
            "vcov": fitted_raw.cov_params(),
            "summary": {"tidy": tidy_df, "glance": glance_dict}
        }
    
    trial_seq.outcome_model["fitted"] = fitted
    return trial_seq


class RobustResults:
    def __init__(self, results, robust_cov):
        self.results = results
        self._robust_cov = robust_cov

    @property
    def params(self):
        return self.results.params

    def cov_params(self):
        return self._robust_cov


def cluster_robust_fit(model, data, cluster_col="id"):
    results = model.fit()
    if cluster_col in data.columns and hasattr(results, "get_robustcov_results"):
        cluster_ids = data[cluster_col]
        results_robust = results.get_robustcov_results(
            cov_type="cluster", 
            groups=cluster_ids,
            use_correction=True,
        )
        return results_robust
    else:
        return results  


class StatsGLMLogit:
    def __init__(self, save_path: Optional[str] = None, robust: bool = True):
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.robust = robust  # NEW flag

    def fit_weights_model(self, data: pd.DataFrame, formula: str, label: str) -> Dict[str, Any]:
        model = smf.glm(formula=formula, data=data, family=sm.families.Binomial())
        results = model.fit()
        fitted_probs = results.fittedvalues
        tidy_df = tidy_statmodels_result(results)
        glance_dict = glance_statmodels_result(results)
        file_path = None
        if self.save_path:
            file_path = os.path.join(self.save_path, f"weights_model_{label.replace(' ', '_')}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(results, f)
        summary_dict = {"tidy": tidy_df, "glance": glance_dict}
        if file_path:
            summary_dict["save_path"] = file_path
        return {"label": label, "summary": summary_dict, "fitted": fitted_probs.values}

    def fit_outcome_model(self, data: pd.DataFrame, formula: str, weights: Optional[np.ndarray] = None, cluster_col: str = "id") -> Dict[str, Any]:
        if weights is None:
            weights = np.ones(len(data), dtype=float)
        data = data.copy()
        data["_weights_"] = weights
        base_model = smf.glm(
            formula=formula,
            data=data,
            family=sm.families.Binomial(),
            freq_weights=data["_weights_"],
            eval_env=0  # Forces patsy to use the global environment.
        )
        results = base_model.fit()
        # Use robust adjustment only if robust flag is True.
        if self.robust and cluster_col in data.columns and hasattr(results, "get_robustcov_results"):
            cluster_ids = data[cluster_col]
            results_robust = results.get_robustcov_results(
                cov_type="cluster", 
                groups=cluster_ids,
                use_correction=True,
            )
        else:
            results_robust = results  # No robust adjustment; AIC and logLik are available.
        tidy_df = tidy_statmodels_result(results_robust)
        glance_dict = glance_statmodels_result(results_robust)
        file_path = None
        if self.save_path:
            file_path = os.path.join(self.save_path, "glm_logit_outcome_model.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(results_robust, f)
        summary_dict = {"tidy": tidy_df, "glance": glance_dict}
        if file_path:
            summary_dict["save_path"] = file_path
        # IMPORTANT: include the formula in the returned model dictionary.
        model_dict = {"params": results_robust.params, "cov_params": results_robust.cov_params(), "formula": formula}
        return {"model": model_dict, "vcov": results_robust.cov_params(), "summary": summary_dict}



def _build_design_matrix(newdata, model_terms):
    """
    Build a design matrix from newdata given a list of model_terms.
    Terms that are wrapped in I(...) are evaluated using pandas.eval.
    """
    X_list = []
    for term in model_terms:
        term_clean = term.strip()
        if term_clean.lower() == "intercept":
            X_list.append(np.ones((newdata.shape[0], 1)))
        elif term_clean.startswith("I(") and term_clean.endswith(")"):
            expr = term_clean[2:-1]
            col = pd.eval(expr, local_dict=newdata.to_dict(orient="series"))
            X_list.append(col.values.reshape(-1, 1))
        else:
            if term_clean in newdata.columns:
                X_list.append(newdata[term_clean].values.reshape(-1, 1))
            else:
                raise ValueError(f"Term '{term_clean}' not found in newdata columns.")
    return np.hstack(X_list)


def check_newdata(newdata, model, predict_times):
    """
    Ensure newdata contains the variables required by the outcome model.
    Assumes model["formula"] is a string like "outcome ~ var1 + var2 + ...".
    Keeps only baseline rows (followup_time == 0).
    """
    formula = model["formula"] if isinstance(model, dict) else model.formula
    if "~" in formula:
        _, rhs = formula.split("~", 1)
        # Split on plus signs and remove extra whitespace
        required_vars = [v.strip() for v in rhs.split("+")]
    else:
        required_vars = []
    # Exclude terms that involve transformations (contain '(') and the intercept "1"
    required_vars = [v for v in required_vars if "(" not in v and v != "1"]
    missing = [v for v in required_vars if v not in newdata.columns]
    if missing:
        raise ValueError("Missing required variables in newdata: " + ", ".join(missing))
    return newdata[newdata["followup_time"] == 0].copy()


def calculate_survival(p_mat):
    # Compute subject-specific survival curves (each row is monotonic)
    surv = np.cumprod(1 - p_mat, axis=1)
    # Compute the log survival curves
    log_surv = np.log(surv)
    # Average the log survival across subjects
    avg_log_surv = np.mean(log_surv, axis=0)
    # Exponentiate to get the overall survival curve
    return np.exp(avg_log_surv)


def calculate_cum_inc(p_mat):
    """
    Cumulative incidence is defined as 1 minus survival.
    """
    return 1 - calculate_survival(p_mat)


def calculate_predictions(newdata, model, treatment_values, pred_fun, coefs_mat, matrix_n_col, design_matrix_builder, n_subjects):
    """
    For each treatment scenario, build a design matrix from newdata (which should be baseline data replicated
    for each follow-up time). Then, for each draw in coefs_mat, compute the linear predictor, apply the logistic
    link, reshape the results into (n_subjects, matrix_n_col), and average over subjects to get the predicted curve.
    
    Parameters:
      newdata : DataFrame (already replicated; its number of rows should equal n_subjects * matrix_n_col)
      model : dict containing "params" (pd.Series)
      treatment_values : dict mapping scenario names to treatment values
      pred_fun : function (calculate_survival or calculate_cum_inc)
      coefs_mat : array of shape (samples+1, n_params)
      matrix_n_col : int, number of follow-up times (len(predict_times))
      design_matrix_builder : callable to build the design matrix
      n_subjects : int, number of baseline subjects
      
    Returns:
      dict mapping scenario names to prediction matrices of shape (matrix_n_col, samples+1)
    """
    model_terms = list(model["params"].index)
    results = {}
    for key, trt_value in treatment_values.items():
        df_temp = newdata.copy()
        df_temp["assigned_treatment"] = trt_value
        # Build design matrix; X will have shape (n_subjects * matrix_n_col, n_params)
        X = design_matrix_builder(df_temp, model_terms)
        pred_matrix = np.zeros((matrix_n_col, coefs_mat.shape[0]))
        for i in range(coefs_mat.shape[0]):
            lp = X.dot(coefs_mat[i, :])  # shape: (n_subjects * matrix_n_col,)
            p_vec = 1.0 / (1.0 + np.exp(-lp))
            # Reshape p_vec into (n_subjects, matrix_n_col)
            p_mat = p_vec.reshape(n_subjects, matrix_n_col)
            # Average across subjects for each follow-up time
            curve = pred_fun(p_mat)
            pred_matrix[:, i] = curve
        results[key] = pred_matrix
    return results


def predict_outcome(trial_seq, newdata, predict_times, conf_int=True, samples=100,
                    type_pred="survival", treatment_col="assigned_treatment",
                    design_matrix_builder=_build_design_matrix):
    """
    Predict marginal survival or cumulative incidence from the fitted outcome model.
    ...
    """
    outcome_fitted = trial_seq.outcome_model["fitted"]
    model = outcome_fitted["model"]
    params = model["params"]
    cov_mat = model["cov_params"]

    # 1. Build coefficient matrix (point estimates + simulation draws)
    coefs_list = [params.values]
    if conf_int:
        draws = multivariate_normal.rvs(mean=params.values, cov=cov_mat, size=samples)
        if draws.ndim == 1:
            draws = draws.reshape(1, -1)
        coefs_list.append(draws)
    coefs_mat = np.vstack(coefs_list)

    # 2. Filter newdata to baseline (followup_time == 0), same as R
    newdata = check_newdata(newdata, model, predict_times)
    newdata = newdata.loc[newdata["followup_time"] == 0].copy()

    baseline_n = newdata.shape[0]
    matrix_n_col = len(predict_times)

    # 4. Replicate each baseline row across predict_times
    newdata_expanded = pd.DataFrame(
        np.repeat(newdata.values, matrix_n_col, axis=0),
        columns=newdata.columns
    )
    newdata_expanded["followup_time"] = np.tile(np.array(predict_times), baseline_n)

    # 5. Choose whether to compute survival or cum_inc
    if type_pred == "survival":
        pred_fun = calculate_survival
    elif type_pred == "cum_inc":
        pred_fun = calculate_cum_inc
    else:
        raise ValueError("type must be either 'survival' or 'cum_inc'.")

    # 6. Generate predictions for treatment=0 and treatment=1
    treatment_values = {"assigned_treatment_0": 0, "assigned_treatment_1": 1}
    pred_dict = calculate_predictions(
        newdata_expanded, model, treatment_values, pred_fun,
        coefs_mat, matrix_n_col, design_matrix_builder, n_subjects=baseline_n
    )

    # 7. Compute difference curve
    #    (If you want "treatment - control", flip the subtraction)
    pred_dict["difference"] = pred_dict["assigned_treatment_0"] - pred_dict["assigned_treatment_1"]

    # 8. Build output DataFrames with optional confidence intervals
    result = {}
    for key, pred_matrix in pred_dict.items():
        point_est = pred_matrix[:, 0]
        if conf_int and pred_matrix.shape[1] > 1:
            lower = np.percentile(pred_matrix[:, 1:], 2.5, axis=1)
            upper = np.percentile(pred_matrix[:, 1:], 97.5, axis=1)
            df = pd.DataFrame({
                "followup_time": predict_times,
                key: point_est,
                "2.5%": lower,
                "97.5%": upper
            })
        else:
            df = pd.DataFrame({
                "followup_time": predict_times,
                key: point_est
            })
        result[key] = df
    return result


def tidy_statmodels_result(results) -> pd.DataFrame:
    """
    Returns a tidy summary DataFrame similar to broom::tidy() in R.
    """
    params = results.params
    cov = results.cov_params()
    se = np.sqrt(np.diag(cov))
    zvals = params / se
    pvals = 2 * (1 - norm.cdf(np.abs(zvals)))
    conf_int = np.column_stack((params - 1.96 * se, params + 1.96 * se))
    tidy_df = pd.DataFrame({
        "term": params.index,
        "estimate": params.values,
        "std_error": se,
        "z_value": zvals,
        "p_value": pvals,
        "conf_low": conf_int[:, 0],
        "conf_high": conf_int[:, 1]
    })
    return tidy_df


def glance_statmodels_result(results) -> dict:
    """
    Returns overall fit metrics similar to broom::glance() in R.
    For robust estimation, metrics like AIC or logLik may be NaN.
    """
    out = {
        "nobs": results.nobs,
        "df_model": results.df_model,
        "df_resid": results.df_resid,
    }
    out["aic"] = getattr(results, "aic", np.nan)
    out["deviance"] = getattr(results, "deviance", np.nan)
    out["logLik"] = getattr(results, "llf", np.nan)
    return out

    
def print_msm_summary(trial_seq):
    """
    Print a summary of the fitted marginal structural model.
    """
    outcome_model = trial_seq.outcome_model
    if "fitted" not in outcome_model or not outcome_model["fitted"]:
        print("No fitted outcome model found. Please run fit_msm() first.")
        return

    fitted = outcome_model["fitted"]
    formula = outcome_model.get("formula", "No formula found")
    tidy_df = fitted["summary"]["tidy"]
    glance = fitted["summary"]["glance"]

    print("Outcome model formula:")
    print(f" - {formula}\n")
    print("Model Summary:")
    for i, row in tidy_df.iterrows():
        term = row['term']
        estimate = row['estimate']
        std_error = row['std_error']
        statistic = row['z_value']
        p_value = row['p_value']
        conf_low = row['conf_low']
        conf_high = row['conf_high']
        print(f" - {term:20s} Estimate: {estimate:10.4f} Std.Error: {std_error:10.4f} "
              f"Statistic: {statistic:10.4f} p-value: {p_value:10.4f} "
              f"Conf.Low: {conf_low:10.4f} Conf.High: {conf_high:10.4f}")
    print("\nModel Fit Metrics:")
    for key, value in glance.items():
        print(f" - {key}: {value}")
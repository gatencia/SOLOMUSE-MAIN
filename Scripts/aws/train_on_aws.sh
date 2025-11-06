#!/bin/bash

# ... (previous content of the file)

# Inside the SMO tuning loop
if [ -z "$best_cv" ] || awk -v a="$cv_acc" -v b="$best_cv" 'BEGIN{exit !(a > b)}'; then
  # ... rest of the loop

# Inside the LWL tuning loop
if [ -z "$best_rmse" ] || awk -v a="$cv_rmse" -v b="$best_rmse" 'BEGIN{exit !(a < b)}'; then
  # ... rest of the loop

# Inside the RandomForest tuning loop
if [ -z "$best_rmse" ] || awk -v a="$cv_rmse" -v b="$best_rmse" 'BEGIN{exit !(a < b)}'; then
  # ... rest of the loop
# initialize sentinels if still empty (prevents awk syntax errors like 'a > ')
: "${best_cv:=}"
: "${best_rmse:=}"

# SMO sentinels: treat empty best as -inf accuracy
if [ -z "$best_cv" ]; then best_cv="-1"; fi

# Regression sentinels: treat empty best as +inf RMSE
if [ -z "$best_rmse" ]; then best_rmse="1e100"; fi

# safety: if no best found (e.g., parse failure), fall back to defaults
if [ -z "$best_C" ] || [ -z "$best_G" ]; then
  best_C=1
  best_G=0.01
fi
if [ -z "$best_K" ]; then
  best_K=5
fi
if [ -z "$best_N" ]; then
  best_N=100
fi

# Evaluate the best on TEST
# ... rest of the file

# When invoking SMO with RBF kernel, pass kernel option as:
# -K "weka.classifiers.functions.supportVector.RBFKernel -G $G"

# For LWL and RF, ensure results are written as a single CSV line:
# Example for LWL:
# printf "%s,%s,%q,%s,%s\n" "$fold" "$cv_rmse" "$best_opts" "$test_rmse" "$test_mae" >> results_reg_lwl.csv
# Example for RF:
# printf "%s,%s,%q,%s,%s\n" "$fold" "$cv_rmse" "$best_opts" "$test_rmse" "$test_mae" >> results_reg_rf.csv
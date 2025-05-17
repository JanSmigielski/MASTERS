import pandas as pd
import os
from models import simple_model, complex_model, performance_metrics, factor_stats

model_funcs = {
    'simple': simple_model,
    'complex': complex_model
}

predictors = pd.read_excel("predictors.xlsx")
predictors.set_index('date', inplace=True)

factor_data_raw = pd.read_excel("us_factors.xlsx", sheet_name=None)


factor_data = {}
for sheet_name, df in factor_data_raw.items():
    df = df.rename(columns={'ret': sheet_name})
    df.set_index('date', inplace=True)
    factor_data[sheet_name] = df

os.makedirs("results", exist_ok=True)
summary_records = []
factor_records = []

for factor_name, df in factor_data.items():

    target = df[factor_name]
    aligned_predictors = predictors.loc[predictors.index.intersection(target.index)]
    aligned_target = target.loc[aligned_predictors.index]
    
    factor_metrics = factor_stats(aligned_target)
    factor_metrics['factor'] = factor_name
    factor_records.append(factor_metrics)

    for model_name, model_func in model_funcs.items():
        print(f"Running {model_name} model for {factor_name}...")
        result_df = model_func(aligned_predictors, aligned_target, factor_name)
        metrics = performance_metrics(result_df, target_name=factor_name)

        out_path = f"results/{factor_name}_{model_name}.xlsx"
        with pd.ExcelWriter(out_path) as writer:
            result_df.to_excel(writer, sheet_name="predictions")
            pd.DataFrame([metrics]).to_excel(writer, sheet_name="performance")

        metrics['factor'] = factor_name
        metrics['model'] = model_name
        summary_records.append(metrics)

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv("results/summary_metrics.csv", index=False)

factor_df = pd.DataFrame(factor_records)
factor_df.to_csv("results/factor_metrics.csv", index=False)

print("Pipeline complete. Results saved in /results/")


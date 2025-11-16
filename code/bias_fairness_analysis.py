"""
Bias/Fairness Analysis for Chicago Crime Prediction
This module performs fairness analysis on the crime prediction model.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, stddev, to_timestamp, year, weekofyear, sum as spark_sum
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Initialize Spark
spark = SparkSession.builder.appName("CrimeFairnessAnalysis").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("="*80)
print("BIAS/FAIRNESS ANALYSIS FOR CHICAGO CRIME PREDICTION")
print("="*80)



print("\n[1] Loading Chicago Crime Dataset...")
df = spark.read.csv("hdfs://wolf:9000/user/pin2118/Crimes_-_2001_to_present.csv", header=True)

print(f"Total records loaded: {df.count():,}")

# Define violent crime types
violent_types = [
    "BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE",
    "CRIM SEXUAL ASSAULT", "SEX OFFENSE", "WEAPONS VIOLATION", 
    "CRIMINAL SEXUAL ASSAULT", "KIDNAPPING", "INTIMIDATION", 
    "STALKING", "OFFENSE INVOLVING CHILDREN", "CRIMINAL DAMAGE", "NARCOTICS"
]

print("\n[2] Protected Attributes:")
print("  - District (socioeconomic/demographic proxy)")
print("  - Domestic violence indicator")



print("\n[3] Preprocessing Data...")

# Create violent crime indicator
df = df.withColumn("Violent", col("Primary Type").isin(violent_types).cast("int"))

# Parse timestamp and extract temporal features
df = df.withColumn("Date", to_timestamp("Date", "MM/dd/yyyy hh:mm:ss a"))
df = df.withColumn("Week", weekofyear("Date"))
df = df.withColumn("Year", year("Date"))

# Cast boolean columns
df = df.withColumn("Arrest", col("Arrest").cast("int")) \
       .withColumn("Domestic", col("Domestic").cast("int"))

# Clean District column
df = df.withColumn(
    "District_Clean",
    when(col("District").cast("int").between(1, 25), col("District").cast("int"))
    .otherwise(None)
)



print("\n[4] Detecting Bias in Data...")

# Analyze arrest rates by district
arrest_by_district = df.groupBy("District_Clean").agg(
    count("*").alias("Total_Crimes"),
    spark_sum("Arrest").alias("Total_Arrests"),
    (spark_sum("Arrest") / count("*") * 100).alias("Arrest_Rate_Pct")
).filter(col("District_Clean").isNotNull())

print("\nArrest Rates by District (Top 5):")
arrest_by_district.orderBy(col("Arrest_Rate_Pct").desc()).show(5, truncate=False)

# Analyze violent crime distribution
violent_by_district = df.groupBy("District_Clean").agg(
    count("*").alias("Total_Crimes"),
    spark_sum("Violent").alias("Violent_Crimes"),
    (spark_sum("Violent") / count("*") * 100).alias("Violent_Rate_Pct")
).filter(col("District_Clean").isNotNull())

print("\nViolent Crime Rates by District (Top 5):")
violent_by_district.orderBy(col("Violent_Rate_Pct").desc()).show(5, truncate=False)

print("\n[5] Hypothesis:")
print("  - Districts may have different arrest/crime patterns")
print("  - Model predictions may vary systematically by district")



print("\n[6] Building Crime Prediction Model...")

# Aggregate to Beat-Year-Week level
agg_df = df.groupBy("Beat", "Year", "Week", "District_Clean").agg(
    count("ID").alias("TotalCrimes"),
    spark_sum("Violent").alias("ViolentCrimes"),
    avg("Arrest").alias("ArrestRate"),
    avg("Domestic").alias("DomesticRate")
)

# Compute percent violent crimes
agg_df = agg_df.withColumn(
    "percent_violent_crimes",
    when(col("TotalCrimes") != 0, col("ViolentCrimes") / col("TotalCrimes")).otherwise(0)
)

agg_df = agg_df.fillna(0)

# Create lag features
windowSpec = Window.partitionBy("Beat").orderBy("Year", "Week")
agg_df = agg_df.withColumn("total_last_week", F.lag("TotalCrimes", 1).over(windowSpec)).fillna(0)
agg_df = agg_df.withColumn("violent_crimes_next_week", F.lead("ViolentCrimes", 1).over(windowSpec))
agg_df = agg_df.filter(col("violent_crimes_next_week").isNotNull())

# Create district risk groups for fairness analysis
agg_df = agg_df.withColumn(
    "DistrictGroup",
    when(col("District_Clean") <= 8, "North")
    .when(col("District_Clean") <= 16, "Central")
    .otherwise("South")
)

# Train/Test split
agg_df = agg_df.withColumn("time_index", F.row_number().over(windowSpec) - 1)
max_index = agg_df.agg(F.max("time_index")).first()[0]
cutoff = int(max_index * 0.8)
train_df = agg_df.filter(col("time_index") <= cutoff)
test_df = agg_df.filter(col("time_index") > cutoff)

print(f"Training samples: {train_df.count():,}")
print(f"Testing samples: {test_df.count():,}")

# Build ML Pipeline
indexer = StringIndexer(
    inputCols=["Beat", "Year", "Week"], 
    outputCols=["BeatIdx", "YearIdx", "WeekIdx"],
    handleInvalid="keep"
)
encoder = OneHotEncoder(
    inputCols=["BeatIdx", "YearIdx", "WeekIdx"], 
    outputCols=["BeatVec", "YearVec", "WeekVec"]
)

feature_cols = [
    "BeatVec", "YearVec", "WeekVec", "TotalCrimes", "ArrestRate", 
    "DomesticRate", "percent_violent_crimes", "total_last_week"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
gbt = GBTRegressor(labelCol="violent_crimes_next_week", featuresCol="features", maxIter=100)

pipeline = Pipeline(stages=[indexer, encoder, assembler, gbt])

print("Training model...")
model = pipeline.fit(train_df)
predictions = test_df.transform(model)

# Overall model performance
evaluator_rmse = RegressionEvaluator(labelCol="violent_crimes_next_week", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="violent_crimes_next_week", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="violent_crimes_next_week", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("\n" + "="*80)
print("OVERALL MODEL PERFORMANCE")
print("="*80)
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print("="*80)


print("\n[7] Fairness Metrics Analysis...")
print("="*80)

def compute_fairness_metrics(predictions_df, group_col, group_name):
    """Compute fairness metrics for different groups"""
    print(f"\n{group_name} Analysis")
    print("-" * 80)
    
    metrics_by_group = predictions_df.groupBy(group_col).agg(
        count("*").alias("count"),
        avg("violent_crimes_next_week").alias("actual_avg"),
        avg("prediction").alias("predicted_avg"),
        avg(F.abs(col("violent_crimes_next_week") - col("prediction"))).alias("mae"),
        avg((col("prediction") - col("violent_crimes_next_week"))).alias("mean_error")
    )
    
    metrics_pd = metrics_by_group.toPandas()
    print(metrics_pd.to_string(index=False))
    
    if len(metrics_pd) > 1:
        max_pred = metrics_pd['predicted_avg'].max()
        min_pred = metrics_pd['predicted_avg'].min()
        disparate_impact = min_pred / max_pred if max_pred > 0 else 0
        
        print(f"\nDisparate Impact Ratio: {disparate_impact:.4f}")
        print(f"  (Values < 0.8 suggest potential bias)")
        
        error_diff = metrics_pd['mean_error'].max() - metrics_pd['mean_error'].min()
        print(f"\nMean Error Difference: {error_diff:.4f}")
    
    print("-" * 80)
    return metrics_pd

# Analyze by District Group
district_metrics = compute_fairness_metrics(predictions, "DistrictGroup", "DISTRICT GROUP (Geographic)")


print("\n[8] Generating Visualizations...")

viz_data = predictions.select(
    "violent_crimes_next_week",
    "prediction",
    "DistrictGroup"
).toPandas()

import os
os.makedirs("/nfs/home/pin2118/hw3_folder/fairness_viz", exist_ok=True)

# Prediction Error by District Group
viz_data['error'] = viz_data['prediction'] - viz_data['violent_crimes_next_week']

plt.figure(figsize=(10, 6))
sns.boxplot(data=viz_data, x='DistrictGroup', y='error', order=["North", "Central", "South"])
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.title('Prediction Error by District Group')
plt.xlabel('District Group')
plt.ylabel('Prediction Error (Predicted - Actual)')
plt.tight_layout()
plt.savefig("/nfs/home/pin2118/hw3_folder/fairness_viz/prediction_error_by_district.png", dpi=300)
print("✓ Saved: prediction_error_by_district.png")

# Actual vs Predicted
plt.figure(figsize=(8, 6))
for group in ["North", "Central", "South"]:
    group_data = viz_data[viz_data['DistrictGroup'] == group]
    plt.scatter(group_data['violent_crimes_next_week'], group_data['prediction'], 
                alpha=0.3, label=group, s=10)
plt.plot([0, viz_data['violent_crimes_next_week'].max()], 
         [0, viz_data['violent_crimes_next_week'].max()], 
         'r--', label='Perfect Prediction')
plt.xlabel('Actual Violent Crimes')
plt.ylabel('Predicted Violent Crimes')
plt.title('Predictions by District Group')
plt.legend()
plt.tight_layout()
plt.savefig("/nfs/home/pin2118/hw3_folder/fairness_viz/actual_vs_predicted.png", dpi=300)
print("✓ Saved: actual_vs_predicted.png")


print("\n" + "="*80)
print("FAIRNESS ANALYSIS RESULTS")
print("="*80)
print("\nKEY FINDINGS:")
print("  - Variation in prediction accuracy across district groups")
print("  - Potential systematic bias in certain geographic areas")
print("\nMITIGATION STRATEGIES:")
print("  1. Re-weight training samples by district")
print("  2. Add fairness constraints to model optimization")
print("  3. Post-process calibration by group")
print("  4. Regular fairness audits")
print("="*80)

# Save results
district_metrics.to_csv("/nfs/home/pin2118/hw3_folder/fairness_metrics_district.csv", index=False)

with open("/nfs/home/pin2118/hw3_folder/fairness_analysis_report.txt", 'w') as f:
    f.write("BIAS/FAIRNESS ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    f.write("Protected Attributes: District (geographic/demographic proxy)\n")
    f.write("Fairness Metrics: Disparate Impact, Mean Error by Group\n")
    f.write("Mitigation: Re-weighting, fairness constraints, calibration\n")

print("\n✓ Results saved to: /nfs/home/pin2118/hw3_folder/")
print("  - fairness_metrics_district.csv")
print("  - fairness_analysis_report.txt")
print("  - fairness_viz/ (visualizations)")

spark.stop()
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    to_timestamp, month, col, hour, dayofweek, year,
    when, count, sum as spark_sum
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

print("="*80)
print("COMPREHENSIVE TEMPORAL & DISTRIBUTION ANALYSIS")
print("Chicago Crime Data (2001-present)")
print("="*80)

# Create SparkSession
spark = SparkSession.builder \
    .appName("TemporalDistributionAnalysis") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load data
print("\n[1] Loading Crime Dataset...")
df = spark.read.csv("hdfs://wolf:9000/user/pin2118/Crimes_-_2001_to_present.csv", header=True)

total_crimes = df.count()
print(f"✓ Total crimes loaded: {total_crimes:,}")

# Parse Date column into Timestamp
df = df.withColumn("ts", to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a"))

# Define violent crime types
violent_types = [
    "BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE",
    "CRIM SEXUAL ASSAULT", "SEX OFFENSE", "WEAPONS VIOLATION", 
    "CRIMINAL SEXUAL ASSAULT", "KIDNAPPING", "INTIMIDATION"
]

# Add violent crime indicator
df = df.withColumn("is_violent", 
    when(col("Primary Type").isin(violent_types), 1).otherwise(0)
)

# Register as temp SQL view
df.createOrReplaceTempView("crime_data")

# Create output directory
os.makedirs("/nfs/home/pin2118/hw3_folder/temporal_analysis", exist_ok=True)

print("\n[2] Analyzing Crime Distributions...")

# --- District Distribution ---
print("  → Crime distribution by District")
district_result = spark.sql("""
    SELECT 
      District,
      COUNT(*) AS Crime_Count
    FROM crime_data
    WHERE District IS NOT NULL AND District != ''
    GROUP BY District
    ORDER BY CAST(District AS INT)
""")

district_data = district_result.toPandas()

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.bar(district_data['District'].astype(str), district_data['Crime_Count'], color='steelblue')
plt.xlabel("District")
plt.ylabel("Number of Crimes")
plt.title("Crime Distribution by District")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# --- Primary Type Distribution (Top 15) ---
print("  → Crime distribution by Primary Type")
type_result = spark.sql("""
    SELECT 
      `Primary Type`,
      COUNT(*) AS Crime_Count
    FROM crime_data
    WHERE `Primary Type` IS NOT NULL
    GROUP BY `Primary Type`
    ORDER BY Crime_Count DESC
    LIMIT 15
""")

type_data = type_result.toPandas()

plt.subplot(1, 2, 2)
plt.barh(type_data['Primary Type'], type_data['Crime_Count'], color='coral')
plt.xlabel("Number of Crimes")
plt.ylabel("Crime Type")
plt.title("Top 15 Crime Types")
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("/nfs/home/pin2118/hw3_folder/temporal_analysis/distribution_analysis.png", dpi=300)
print("  ✓ Saved: distribution_analysis.png")

# Save distribution data
district_result.write.csv(
    "hdfs://wolf:9000/user/pin2118/hw3_folder/temporal_analysis/district_distribution.csv",
    header=True, mode="overwrite"
)
type_result.write.csv(
    "hdfs://wolf:9000/user/pin2118/hw3_folder/temporal_analysis/crime_type_distribution.csv",
    header=True, mode="overwrite"
)


print("\n[3] Analyzing Temporal Patterns...")

# --- Monthly Pattern ---
print("  → Monthly crime patterns")
monthly_result = spark.sql("""
    SELECT 
      MONTH(ts) AS Crime_Month,
      COUNT(*) AS Crime_Count,
      SUM(is_violent) AS Violent_Count,
      COUNT(*) - SUM(is_violent) AS NonViolent_Count
    FROM crime_data
    WHERE ts IS NOT NULL
    GROUP BY MONTH(ts)
    ORDER BY MONTH(ts)
""")

monthly_data = monthly_result.toPandas()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# --- Hourly Pattern ---
print("  → Hourly crime patterns")
hourly_result = spark.sql("""
    SELECT 
      HOUR(ts) AS Crime_Hour,
      COUNT(*) AS Crime_Count,
      SUM(is_violent) AS Violent_Count
    FROM crime_data
    WHERE ts IS NOT NULL
    GROUP BY HOUR(ts)
    ORDER BY HOUR(ts)
""")

hourly_data = hourly_result.toPandas()

# --- Day of Week Pattern ---
print("  → Day of week crime patterns")
dow_result = spark.sql("""
    SELECT 
      DAYOFWEEK(ts) AS Day_of_Week,
      COUNT(*) AS Crime_Count,
      SUM(is_violent) AS Violent_Count
    FROM crime_data
    WHERE ts IS NOT NULL
    GROUP BY DAYOFWEEK(ts)
    ORDER BY DAYOFWEEK(ts)
""")

dow_data = dow_result.toPandas()
day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

# --- Yearly Trend ---
print("  → Yearly crime trends")
yearly_result = spark.sql("""
    SELECT 
      YEAR(ts) AS Crime_Year,
      COUNT(*) AS Crime_Count,
      SUM(is_violent) AS Violent_Count
    FROM crime_data
    WHERE ts IS NOT NULL AND YEAR(ts) >= 2001
    GROUP BY YEAR(ts)
    ORDER BY YEAR(ts)
""")

yearly_data = yearly_result.toPandas()

# Plot temporal patterns
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Monthly
axes[0, 0].bar(range(1, 13), monthly_data['Crime_Count'], color='skyblue', alpha=0.7)
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Number of Crimes")
axes[0, 0].set_title("Crime Events by Month")
axes[0, 0].set_xticks(range(1, 13))
axes[0, 0].set_xticklabels(month_names, rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# Hourly
axes[0, 1].plot(hourly_data['Crime_Hour'], hourly_data['Crime_Count'], 
                marker='o', color='green', linewidth=2)
axes[0, 1].set_xlabel("Hour of Day")
axes[0, 1].set_ylabel("Number of Crimes")
axes[0, 1].set_title("Crime Events by Hour")
axes[0, 1].set_xticks(range(0, 24, 2))
axes[0, 1].grid(alpha=0.3)

# Day of Week
axes[1, 0].bar(range(1, 8), dow_data['Crime_Count'], color='orange', alpha=0.7)
axes[1, 0].set_xlabel("Day of Week")
axes[1, 0].set_ylabel("Number of Crimes")
axes[1, 0].set_title("Crime Events by Day of Week")
axes[1, 0].set_xticks(range(1, 8))
axes[1, 0].set_xticklabels(day_names, rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# Yearly Trend
axes[1, 1].plot(yearly_data['Crime_Year'], yearly_data['Crime_Count'], 
                marker='s', color='red', linewidth=2)
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("Number of Crimes")
axes[1, 1].set_title("Crime Trend Over Years")
axes[1, 1].grid(alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("/nfs/home/pin2118/hw3_folder/temporal_analysis/temporal_patterns.png", dpi=300)
print("  ✓ Saved: temporal_patterns.png")


print("\n[4] Performing Cross-Analysis...")

# Violent vs Non-Violent by Month
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
x = range(1, 13)
width = 0.35
plt.bar([i - width/2 for i in x], monthly_data['Violent_Count'], 
        width, label='Violent', color='darkred', alpha=0.7)
plt.bar([i + width/2 for i in x], monthly_data['NonViolent_Count'], 
        width, label='Non-Violent', color='steelblue', alpha=0.7)
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.title("Violent vs Non-Violent Crimes by Month")
plt.xticks(x, month_names, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Violent crime percentage by hour
plt.subplot(1, 2, 2)
hourly_data['Violent_Pct'] = (hourly_data['Violent_Count'] / hourly_data['Crime_Count']) * 100
plt.plot(hourly_data['Crime_Hour'], hourly_data['Violent_Pct'], 
         marker='o', color='darkred', linewidth=2)
plt.xlabel("Hour of Day")
plt.ylabel("Violent Crime %")
plt.title("Percentage of Violent Crimes by Hour")
plt.xticks(range(0, 24, 2))
plt.grid(alpha=0.3)
plt.axhline(y=hourly_data['Violent_Pct'].mean(), color='gray', 
            linestyle='--', label='Average')
plt.legend()

plt.tight_layout()
plt.savefig("/nfs/home/pin2118/hw3_folder/temporal_analysis/violent_nonviolent_analysis.png", dpi=300)
print("  ✓ Saved: violent_nonviolent_analysis.png")


print("\n[5] Generating Statistical Summary...")

# Calculate statistics
total_violent = df.filter(col("is_violent") == 1).count()
total_nonviolent = total_crimes - total_violent
violent_pct = (total_violent / total_crimes) * 100

peak_month = monthly_data.loc[monthly_data['Crime_Count'].idxmax(), 'Crime_Month']
peak_month_name = month_names[int(peak_month) - 1]

peak_hour = hourly_data.loc[hourly_data['Crime_Count'].idxmax(), 'Crime_Hour']
peak_day = dow_data.loc[dow_data['Crime_Count'].idxmax(), 'Day_of_Week']
peak_day_name = day_names[int(peak_day) - 1]

# Top crime type
top_crime = type_data.iloc[0]['Primary Type']
top_crime_count = type_data.iloc[0]['Crime_Count']

# Yearly trend (first vs last year)
first_year = yearly_data.iloc[0]
last_year = yearly_data.iloc[-1]
crime_change = ((last_year['Crime_Count'] - first_year['Crime_Count']) / first_year['Crime_Count']) * 100

# Create summary report
summary = f"""
{'='*80}
STATISTICAL SUMMARY - CHICAGO CRIME ANALYSIS
{'='*80}

OVERALL STATISTICS:
------------------
Total Crimes:                {total_crimes:,}
Violent Crimes:              {total_violent:,} ({violent_pct:.2f}%)
Non-Violent Crimes:          {total_nonviolent:,} ({100-violent_pct:.2f}%)

TEMPORAL PEAKS:
--------------
Peak Crime Month:            {peak_month_name} ({monthly_data.loc[monthly_data['Crime_Count'].idxmax(), 'Crime_Count']:,} crimes)
Peak Crime Hour:             {int(peak_hour):02d}:00 ({hourly_data.loc[hourly_data['Crime_Count'].idxmax(), 'Crime_Count']:,} crimes)
Peak Crime Day:              {peak_day_name} ({dow_data.loc[dow_data['Crime_Count'].idxmax(), 'Crime_Count']:,} crimes)

CRIME TYPE ANALYSIS:
-------------------
Most Common Crime Type:      {top_crime}
Count:                       {top_crime_count:,} crimes
Percentage of Total:         {(top_crime_count/total_crimes)*100:.2f}%

YEARLY TREND:
------------
First Year ({int(first_year['Crime_Year'])}):         {int(first_year['Crime_Count']):,} crimes
Last Year ({int(last_year['Crime_Year'])}):          {int(last_year['Crime_Count']):,} crimes
Change:                      {crime_change:+.2f}%

VIOLENT CRIME PATTERNS:
----------------------
Highest Violent % Hour:      {int(hourly_data.loc[hourly_data['Violent_Pct'].idxmax(), 'Crime_Hour']):02d}:00 ({hourly_data['Violent_Pct'].max():.2f}%)
Lowest Violent % Hour:       {int(hourly_data.loc[hourly_data['Violent_Pct'].idxmin(), 'Crime_Hour']):02d}:00 ({hourly_data['Violent_Pct'].min():.2f}%)

{'='*80}
"""

print(summary)

# Save summary to file
with open("/nfs/home/pin2118/hw3_folder/temporal_analysis/statistical_summary.txt", 'w') as f:
    f.write(summary)

# Save to HDFS as well
spark.sparkContext.parallelize([summary]).saveAsTextFile(
    "hdfs://wolf:9000/user/pin2118/hw3_folder/temporal_analysis/summary.txt"
)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nOutputs saved to:")
print("  - /nfs/home/pin2118/hw3_folder/temporal_analysis/")
print("  - hdfs://wolf:9000/user/pin2118/hw3_folder/temporal_analysis/")
print("\nGenerated Files:")
print("  ✓ distribution_analysis.png")
print("  ✓ temporal_patterns.png")
print("  ✓ violent_nonviolent_analysis.png")
print("  ✓ statistical_summary.txt")
print("  ✓ district_distribution.csv")
print("  ✓ crime_type_distribution.csv")

spark.stop()
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, month, col
import matplotlib.pyplot as plt

# 1. Create SparkSession
spark = SparkSession.builder \
    .appName("Question1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Load the sample CSV
df = spark.read.csv("hdfs://wolf:9000/user/pin2118/Crimes_-_2001_to_present.csv", header=True)


# 3. Parse Date column into Timestamp
df = df.withColumn("ts", to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a"))

df.createOrReplaceTempView("crime_data")

# 6. Run SQL to extract month
result = spark.sql("""
    SELECT 
      MONTH(ts) AS Crime_Month,
      COUNT(Arrest) AS Arrest_Count
    FROM crime_data
    WHERE Arrest is True
    GROUP BY MONTH(ts)
    ORDER BY MONTH(ts)
""")

result.rdd.map(lambda row: f"{row['Crime_Month']}\t{row['Arrest_Count']}") \
    .saveAsTextFile("hdfs://wolf:9000/user/pin2118/hw3_folder/question3_part1_output_txt")


monthly_data = result.collect()

# 7. Convert to X, Y lists for plotting
months = [row['Crime_Month'] for row in monthly_data]
counts = [row['Arrest_Count'] for row in monthly_data]

plt.figure(figsize=(8, 5))
plt.bar(months, counts, color="skyblue")
plt.xlabel("Month")
plt.ylabel("Number of Arrested Events")
plt.title("Arrest Events by Month")
plt.xticks(range(1, 13))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("/nfs/home/pin2118/hw3_folder/question3_arrest_patterns_by_month.png")

print('Question 3 Task 1 Done')


# 6. Run SQL to extract month
result_2 = spark.sql("""
    SELECT 
      HOUR(ts) AS hours_of_day,
      COUNT(Arrest) AS Arrest_Count
    FROM crime_data
    WHERE Arrest is True
    GROUP BY HOUR(ts)
    ORDER BY HOUR(ts)
""")

result_2.rdd.map(lambda row: f"{row['hours_of_day']}\t{row['Arrest_Count']}") \
    .saveAsTextFile("hdfs://wolf:9000/user/pin2118/hw3_folder/question3_part2_output_txt")


hourly_data = result_2.collect()

# 7. Convert to X, Y lists for plotting
hours = [row['hours_of_day'] for row in hourly_data]
counts = [row['Arrest_Count'] for row in hourly_data]

plt.figure(figsize=(8, 5))
plt.bar(hours, counts, color="skyblue")
plt.xlabel("Hours of the Day")
plt.ylabel("Number of Arrested Events")
plt.title("Arrest Events by Hour")
plt.xticks(range(0,24))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("/nfs/home/pin2118/hw3_folder/question3_arrest_patterns_by_hour.png")

print('Question 3 Task 2 Done')

# 6. Run SQL to extract month
result_3 = spark.sql("""
    SELECT 
      DAYOFWEEK(ts) AS day_of_week,
      COUNT(Arrest) AS Arrest_Count
    FROM crime_data
    WHERE Arrest is True
    GROUP BY DAYOFWEEK(ts)
    ORDER BY DAYOFWEEK(ts)
""")

result_3.rdd.map(lambda row: f"{row['day_of_week']}\t{row['Arrest_Count']}") \
    .saveAsTextFile("hdfs://wolf:9000/user/pin2118/hw3_folder/question3_part3_output_txt")


day_of_week_data = result_3.collect()

# 7. Convert to X, Y lists for plotting
weeks = [row['day_of_week'] for row in day_of_week_data]
counts = [row['Arrest_Count'] for row in day_of_week_data]

plt.figure(figsize=(8, 5))
plt.bar(weeks, counts, color="skyblue")
plt.xlabel("Day of Week")
plt.ylabel("Number of Arrested Events")
plt.title("Arrest Events by Day of Week")
plt.xticks(range(1,8),['Monday', 'Tuesday','Wednesday','Thursday','Fridan','Saturday', 'Sunday'])
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("/nfs/home/pin2118/hw3_folder/question3_arrest_patterns_by_day_of_week.png")

print('Question 3 Task 3 Done')


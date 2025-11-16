from pyspark.sql import SparkSession
from datetime import datetime
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat  import Statistics


# 1. Create SparkSession
spark = SparkSession.builder \
    .appName("Question2_Part1") \
    .getOrCreate()

sc = spark.sparkContext

# 2. Load data from HDFS
df = spark.read.csv("hdfs://wolf:9000/user/pin2118/Crimes_-_2001_to_present.csv", header=True)
data = df.rdd.map(lambda row: [str(x) for x in row])  # ensure values are strings

# 3. Get header and remove it
header = data.first()
records = data.filter(lambda row: row != header)

# 4. Parse rows and filter for crimes after 2022
date_threshold = 2019

def parse_line_1(fields):
    try:
        date_str = fields[2]  # Date
        block = fields[3]     # Block
        crime_date = datetime.strptime(date_str, "%m/%d/%Y %I:%M:%S %p")
        return (block, crime_date.year)
    except:
        return (None, None)

block_year = records.map(parse_line_1).filter(lambda x: x[0] and x[1] >= date_threshold)

# 5. Count crimes per block and get top 10
block_counts = block_year.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
top_10_blocks = block_counts.sortBy(lambda x: -x[1]).take(10)

# 6. Convert to RDD and save as text to HDFS
top_10_rdd = sc.parallelize(top_10_blocks)
top_10_rdd.map(lambda x: f"{x[0]}\t{x[1]}") \
    .saveAsTextFile("hdfs://wolf:9000/user/pin2118/hw3_folder/question2_part1_top10.txt")

print("✅ Question 2 Part 1 output saved to HDFS.")


def parse_line_2(fields):
    # skip header or malformed
    if fields[0] == "ID" or len(fields) <= 10:
        return None

    try:
        date_str = fields[2]             # 'Date' column
        beat = fields[10]                # 'Beat' column
        crime_year = datetime.strptime(date_str, "%m/%d/%Y %I:%M:%S %p").year
        return (beat, crime_year)
    except:
        return None

# 2) Apply to RDD and filter by year
date_threshold = 2017
beat_year = (
    records
    .map(parse_line_2)
    .filter(lambda x: x is not None and x[1] >= date_threshold)
)


beat_year_count = (
    beat_year
    .map(lambda x: ((x[0], x[1]), 1))
    .reduceByKey(lambda a, b: a + b)
)

years = [2017, 2018, 2019, 2020, 2021, 2022]

beat_grouped = (
    beat_year_count
    .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
    .groupByKey()
    .mapValues(list)
)

beat_vectors = beat_grouped.mapValues(
    lambda yc: [dict(yc).get(y, 0) for y in years]
)

_, mat6x = zip(*beat_vectors.collect())
mat_transposed = list(zip(*mat6x))

year_vectors = sc.parallelize(mat_transposed).map(lambda row: Vectors.dense(row))
year_corr = Statistics.corr(year_vectors, method="pearson")

beat_list = beat_vectors.map(lambda x: x[0]).collect()

correlations = []
for i in range(len(beat_list)):
    for j in range(i + 1, len(beat_list)):
        correlations.append((beat_list[i], beat_list[j], year_corr[i][j]))

filtered_pairs = [
    (b1, b2, corr)
    for (b1, b2, corr) in correlations
    if b1[:-2] == b2[:-2]
]

# 10. Sort filtered pairs by absolute correlation
filtered_pairs_sort = sorted(filtered_pairs, key=lambda x: -abs(x[2]))

# 11. Save result to HDFS
sc.parallelize(filtered_pairs_sort) \
  .map(lambda x: f"{x[0]}\t{x[1]}\t{round(x[2], 4)}") \
  .saveAsTextFile("hdfs://wolf:9000/user/pin2118/hw3_folder/question2_part2_correlated.txt")

print("✅ Question 2 Part 2 done and output saved to HDFS.")



def extract_district_year(fields):
    try:
        date_str = fields[2]
        district_str = fields[11].strip()
        year = datetime.strptime(date_str, "%m/%d/%Y %I:%M:%S %p").year
        district = int(float(district_str))  # handle "6.0", "24.0", etc.
        if 1 <= district <= 25:
            return (str(district), year)
        else:
            return None
    except:
        return None



district_year = records.map(extract_district_year).filter(lambda x: x is not None)

def label_mayor(x):
    district, year = x
    if year >= 2001 and year <= 2011:
        return (district, "Daley")
    elif year < 2019:
        return (district, "Emanuel")
    else:
        return None

labeled = district_year.map(label_mayor).filter(lambda x: x is not None)

# 5. Count crimes per (district, mayor)
district_mayor_counts = labeled.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda a, b: a + b)
reshaped = district_mayor_counts.map(lambda x: (x[0][0], (x[0][1], x[1])))  # (district, (mayor, count))
grouped = reshaped.groupByKey().mapValues(dict)

# 6. Normalize by number of years
years_daley = 11
years_emanuel = 8

district_avg = grouped.map(lambda kv: (
    kv[0],
    kv[1].get("Daley", 0) / years_daley,
    kv[1].get("Emanuel", 0) / years_emanuel
))

# 7. Collect and calculate t-statistic manually
rows = district_avg.collect()
daley = [row[1] for row in rows]
emanuel = [row[2] for row in rows]

diffs = [d - e for d, e in zip(daley, emanuel)]
n = len(diffs)
mean_diff = sum(diffs) / n
sq_diffs = [(d - mean_diff) ** 2 for d in diffs]
std_diff = (sum(sq_diffs) / (n - 1)) ** 0.5
t_stat = mean_diff / (std_diff / (n ** 0.5))

# 8. Save result to HDFS
output_path = "hdfs://wolf:9000/user/pin2118/hw3_folder/question2_part3_tstat.txt"
sc.parallelize([f"T-statistic: {t_stat:.4f}"]).saveAsTextFile(output_path)

print("✅ Question 2 Part 3 completed. T-stat saved to HDFS.")


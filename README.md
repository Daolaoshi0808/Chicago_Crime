# Chicago Crime Prediction with Fairness-Aware ML

**Advanced crime forecasting system using Apache Spark ML Pipelines with comprehensive geospatial and fairness analysis**


---

##  Project Overview

This project implements a production-ready machine learning system for **beat-level violent crime prediction** in Chicago, processing over **8 million crime events** (2001-present). The system combines advanced time-series forecasting with fairness-aware algorithms and multi-scale geospatial analysis.

### Key Features

-  **Beat-level crime prediction** using Gradient Boosted Trees (GBT) regressor
-  **Advanced time-series feature engineering** (lag features, rolling averages, seasonal indicators)
-  **Multi-scale geospatial analysis** (Block → Beat → District hierarchy)
-  **Fairness-aware ML** with bias detection and mitigation strategies
-  **Comprehensive temporal & distribution analysis**
-  **Production-ready Spark ML Pipelines**

---

##  Architecture

```
Chicago Crime Data (8M+ events)
         ↓
    [Apache Spark]
         ↓
    ┌────────────────────────────┐
    │  Data Processing Pipeline  │
    └────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │  1. Temporal Analysis                   │
    │  2. Geospatial Analysis                 │
    │  3. Arrest Pattern Analysis             │
    │  4. Crime Prediction Model              │
    │  5. Bias/Fairness Analysis              │
    └─────────────────────────────────────────┘
         ↓
    [Insights & Predictions]
```

---

## Project Structure

```
chicago-crime-prediction/
│
├── temporal_analysis.py           # Comprehensive temporal & distribution analysis
├── geospatial_analysis.py         # Multi-scale spatial analysis (Block/Beat/District)
├── arrest_pattern_analysis.py     # Arrest patterns by time/location
├── crime_prediction_model.py      # Beat-level violent crime forecasting
├── bias_fairness_analysis.py      # Fairness metrics & bias detection
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── outputs/                        # Generated visualizations & results
```

---

## Quick Start

### Prerequisites

- Apache Spark 3.x
- Python 3.8+
- HDFS access (or modify paths for local filesystem)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chicago-crime-prediction.git
cd chicago-crime-prediction

# Install dependencies
pip install pyspark pandas matplotlib seaborn numpy
```

### Running the Analysis

```bash
# 1. Temporal and distribution analysis
spark-submit temporal_analysis.py

# 2. Geospatial analysis (block/beat/district)
spark-submit geospatial_analysis.py

# 3. Arrest pattern analysis
spark-submit arrest_pattern_analysis.py

# 4. Crime prediction model
spark-submit crime_prediction_model.py

# 5. Bias and fairness analysis
spark-submit bias_fairness_analysis.py
```

---

## Analysis Modules

### 1. Temporal Analysis (`temporal_analysis.py`)

Comprehensive exploratory data analysis of crime patterns over time.

**Features:**
- Monthly, hourly, daily, and yearly crime trends
- Distribution by district and crime type
- Violent vs non-violent crime patterns
- Statistical summaries and peak detection

**Outputs:**
- Distribution visualizations (district, crime type)
- Temporal pattern charts (month/hour/day/year)
- Violent vs non-violent cross-analysis
- Statistical summary report

---

### 2. Geospatial Analysis (`geospatial_analysis.py`)

Multi-scale spatial analysis across Chicago's geographic hierarchy.

**Features:**
- **Block-level analysis**: Top crime hotspots at street level
- **Beat correlation**: Spatial relationships between adjacent police beats
- **District comparison**: Crime patterns across police districts by mayoral administration

**Outputs:**
- Top 10 crime blocks (2019-present)
- Adjacent beat crime correlations
- Daley vs Emanuel administration crime comparison (t-statistic)

---

### 3. Arrest Pattern Analysis (`arrest_pattern_analysis.py`)

Temporal patterns of crime arrests across different time dimensions.

**Features:**
- Arrest rates by month, hour of day, and day of week
- Visualization of arrest patterns
- Seasonal arrest trends

**Outputs:**
- Monthly arrest pattern chart
- Hourly arrest pattern chart
- Day-of-week arrest pattern chart

---

### 4. Crime Prediction Model (`crime_prediction_model.py`)

Production-ready ML pipeline for next-week violent crime forecasting at beat level.

**Features:**
- **Time-series features**: 
  - Lag features (1-3 weeks)
  - Rolling averages (3-4 week windows)
  - Seasonal indicators (summer, winter, holidays)
  - Crime trend (rate of change)
- **Model**: Gradient Boosted Trees (GBT) Regressor
- **Spatial granularity**: Police beat level
- **Prediction horizon**: 1 week ahead

**Model Architecture:**
```
Input Features (21 total):
├── Categorical: Beat, Year, Week (one-hot encoded)
├── Aggregated: TotalCrimes, ArrestRate, DomesticRate, percent_violent_crimes
├── Lag Features: total_last_week, total_2weeks_ago, total_3weeks_ago,
│                  violent_last_week, violent_2weeks_ago
├── Rolling Avg: total_crimes_rolling_avg_3wk, violent_crimes_rolling_avg_3wk,
│                total_crimes_rolling_avg_4wk
├── Seasonal: is_summer, is_winter, is_holiday_season
└── Trend: crime_trend

         ↓
    [GBT Regressor]
         ↓
    violent_crimes_next_week (prediction)
```

**Performance Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)  
- R² Score

---

### 5. Bias & Fairness Analysis (`bias_fairness_analysis.py`)

Comprehensive fairness analysis ensuring equitable predictions across demographics.

**Features:**
- Protected attribute identification (District as demographic proxy)
- Bias detection in arrest rates and crime patterns
- Fairness metrics computation
- Prediction error analysis across geographic groups

**Fairness Metrics:**
- Disparate Impact Ratio
- Mean Absolute Error by group
- Prediction error distributions

**Mitigation Strategies:**
- Sample re-weighting
- Fairness-constrained optimization
- Post-processing calibration
- Regular fairness audits

**Outputs:**
- Fairness metrics by district group
- Prediction error visualizations
- Bias analysis report


##  Technical Details

### Data Pipeline
1. **Data Loading**: 8M+ crime records from HDFS
2. **Preprocessing**: Date parsing, feature engineering, aggregation
3. **Feature Engineering**: 21 features including temporal, spatial, and statistical
4. **Model Training**: 80/20 train-test split with time-based ordering
5. **Evaluation**: Multiple metrics (RMSE, MAE, R²) with fairness analysis

### Technologies Used
- **Apache Spark**: Distributed data processing
- **PySpark ML**: Machine learning pipelines
- **Python**: Data analysis and visualization
- **Matplotlib/Seaborn**: Visualization
- **HDFS**: Distributed file storage

---

## 📝 Dataset

**Source**: Chicago Police Department - Crimes (2001-present)

**Size**: 8M+ records

**Key Fields**:
- ID, Date, Block, Beat, District, Ward
- Primary Type, Description, Arrest, Domestic
- Coordinates (X, Y, Latitude, Longitude)

**Geographic Hierarchy**:
```
Block (Street-level)
  ↓ aggregates to
Beat (Police patrol zones - ~280 beats)
  ↓ aggregates to
District (Police districts - 25 districts)
```

---

## Methodology

### Feature Engineering Strategy

**1. Temporal Features**
- Extract: Year, Month, Week, Hour, Day of Week
- Lag: 1-3 week historical crime counts
- Rolling: 3-4 week moving averages
- Seasonal: Binary indicators for summer, winter, holidays
- Trend: Week-over-week crime rate changes

**2. Aggregation Features**
- Total crimes per beat-week
- Violent crime counts and percentages
- Arrest rates
- Domestic incident rates

**3. Categorical Encoding**
- One-hot encoding for Beat, Year, Week
- Handles high cardinality (280+ beats)

### Fairness Considerations

**Protected Attributes**: District (proxy for socioeconomic/demographic status)

**Analysis Approach**:
1. Identify potential bias sources (policing patterns, reporting rates)
2. Measure prediction disparities across groups
3. Quantify fairness metrics
4. Recommend mitigation strategies



## Acknowledgments

- Chicago Police Department for providing open crime data
- Northwestern University for academic support
- Apache Spark community for excellent documentation


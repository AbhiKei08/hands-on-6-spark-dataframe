# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, min, max, count, month, year

# Create a Spark session
spark = SparkSession.builder \
    .appName("Weather Data Analysis") \
    .getOrCreate()

# Load the dataset
weather_df = spark.read.csv("weather_data.csv", header=True, inferSchema=True)

# Show the first few rows of the DataFrame
weather_df.show()

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_descriptive_stats.csv"
task2_output = output_dir + "task2_extreme_weather.csv"
task3_output = output_dir + "task3_weather_trends.csv"
task4_best_days_output = output_dir + "task4_best_days.csv"
task4_worst_days_output = output_dir + "task4_worst_days.csv"

# ------------------------
# Task 1: Descriptive Statistics for Weather Conditions
# ------------------------
def task1_descriptive_stats(weather_df):
    # Calculate average, minimum, and maximum temperature for each location
    desc_stats = weather_df.groupBy("Location").agg(
        avg("MaxTemp").alias("AvgMaxTemp"),
        avg("MinTemp").alias("AvgMinTemp"),
        min("MaxTemp").alias("MinMaxTemp"),
        max("MaxTemp").alias("MaxMaxTemp"),
        avg("Precipitation").alias("AvgPrecipitation"),
        avg("WindSpeed").alias("AvgWindSpeed")
    ).orderBy(col("AvgMaxTemp").desc())
    
    # Show the result
    desc_stats.show()
    
    # Write the result to a CSV file
    desc_stats.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Identifying Extreme Weather Events
# ------------------------
def task2_extreme_weather(weather_df):
    # Define extreme weather conditions
    extreme_weather = weather_df.filter(
        (col("MaxTemp") > 40) | 
        (col("MinTemp") < -10) | 
        (col("Precipitation") > 50) | 
        (col("WindSpeed") > 50)
    )
    
    # Count the number of extreme weather events for each location
    extreme_weather_count = extreme_weather.groupBy("Location").agg(
        count("*").alias("ExtremeWeatherCount")
    ).orderBy(col("ExtremeWeatherCount").desc())
    
    # Show the result
    extreme_weather.show()
    extreme_weather_count.show()
    
    # Write the result to a CSV file
    extreme_weather.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Analyzing Weather Trends Over Time
# ------------------------
def task3_weather_trends(weather_df):
    # Add month and year columns for time-based analysis
    weather_with_time = weather_df.withColumn("Month", month("Date")).withColumn("Year", year("Date"))
    
    # Select a few cities and calculate monthly averages
    cities = ['City1', 'City2']  # Replace with actual city names
    weather_trend = weather_with_time.filter(col("Location").isin(cities)).groupBy("Location", "Year", "Month").agg(
        avg("MaxTemp").alias("AvgMaxTemp"),
        avg("MinTemp").alias("AvgMinTemp"),
        avg("Precipitation").alias("AvgPrecipitation")
    ).orderBy("Location", "Year", "Month")
    
    # Show the result
    weather_trend.show()
    
    # Write the result to a CSV file
    weather_trend.write.csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Finding the Best and Worst Days for Outdoor Activities
# ------------------------
def task4_best_and_worst_days(weather_df):
    # Define conditions for the best days for outdoor activities
    best_days = weather_df.filter(
        (col("MaxTemp").between(20, 30)) &
        (col("Precipitation") < 5) &
        (col("WindSpeed") < 15)
    )
    
    # Define conditions for the worst days for outdoor activities
    worst_days = weather_df.filter(
        (col("MaxTemp") < 0) | 
        (col("MaxTemp") > 35) | 
        (col("Precipitation") > 30) | 
        (col("WindSpeed") > 40)
    )
    
    # Show the results
    best_days.show()
    worst_days.show()
    
    # Write the best days to a CSV file
    best_days.write.csv(task4_best_days_output, header=True)
    print(f"Best Days output written to {task4_best_days_output}")
    
    # Write the worst days to a CSV file
    worst_days.write.csv(task4_worst_days_output, header=True)
    print(f"Worst Days output written to {task4_worst_days_output}")

# ------------------------
# Call the functions for each task
# ------------------------
task1_descriptive_stats(weather_df)
task2_extreme_weather(weather_df)
task3_weather_trends(weather_df)
task4_best_and_worst_days(weather_df)

# Stop the Spark session
spark.stop()

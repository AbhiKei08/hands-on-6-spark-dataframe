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
    # TODO: Implement the code for Task 1: Basic Descriptive Statistics for Weather Conditions
    # Hint: Use groupBy, agg, avg, min, max, etc.

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    # desc_stats.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Identifying Extreme Weather Events
# ------------------------
def task2_extreme_weather(weather_df):
    # TODO: Implement the code for Task 2: Identifying Extreme Weather Events
    # Hint: Filter based on extreme conditions for MaxTemp, MinTemp, Precipitation, WindSpeed

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    # extreme_weather_count.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Analyzing Weather Trends Over Time
# ------------------------
def task3_weather_trends(weather_df):
    # TODO: Implement the code for Task 3: Analyzing Weather Trends Over Time
    # Hint: Add month, year columns and calculate monthly averages

    # Write the result to a CSV file
    # Uncomment the line below after implementing the logic
    # weather_trend.write.csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Finding the Best and Worst Days for Outdoor Activities
# ------------------------
def task4_best_and_worst_days(weather_df):
    # TODO: Implement the code for Task 4: Finding the Best and Worst Days for Outdoor Activities
    # Hint: Use filters for "best" and "worst" days based on weather conditions

    # Write the best days to a CSV file
    # Uncomment the line below after implementing the logic
    # best_days.write.csv(task4_best_days_output, header=True)
    print(f"Best Days output written to {task4_best_days_output}")

    # Write the worst days to a CSV file
    # Uncomment the line below after implementing the logic
    # worst_days.write.csv(task4_worst_days_output, header=True)
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

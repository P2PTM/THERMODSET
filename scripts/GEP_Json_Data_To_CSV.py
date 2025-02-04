import os
import json
import pandas as pd

raw_data_dir = "../data/raw/"
processed_data_dir = "../data/processed/"

os.makedirs(processed_data_dir, exist_ok=True)

# Mapping of file names to zones and measures
file_groups = {
    "Zone 1 - Jan 23_Jun 23": {
        "humidity": os.path.join(raw_data_dir, "Zone 1 - Humidity - Jan 23_Jun 23"),
        "temperature": os.path.join(raw_data_dir, "Zone 1 - Temperature - Jan 23_Jun 23"),
    },
    "Zone 1 - Jun 23_Dec 23": {
        "humidity": os.path.join(raw_data_dir, "Zone 1 - Humidity - Jun 23_Dec 23"),
        "temperature": os.path.join(raw_data_dir, "Zone 1 - Temperature - Jun 23_Dec 23"),
    },
    "Zone 2 - Jan 23_Jun 23": {
        "humidity": os.path.join(raw_data_dir, "Zone 2 - Humidity - Jan 23_Jun 23"),
        "temperature": os.path.join(raw_data_dir, "Zone 2 - Temperature - Jan 23_Jun 23"),
    },
    "Zone 2 - Jun 23_Dec 23": {
        "humidity": os.path.join(raw_data_dir, "Zone 2 - Humidity - Jun 23_Dec 23"),
        "temperature": os.path.join(raw_data_dir, "Zone 2 - Temperature - Jun 23_Dec 23"),
    },
}

final_data = pd.DataFrame()

for zone_period, files in file_groups.items():
    # Load humidity data
    with open(files["humidity"], "r") as hum_file:
        humidity_data = json.load(hum_file)["data"]
        humidity_df = pd.DataFrame(humidity_data)
        humidity_df = humidity_df.rename(columns={"measures": "humidity"})

    # Load temperature data
    with open(files["temperature"], "r") as temp_file:
        temperature_data = json.load(temp_file)["data"]
        temperature_df = pd.DataFrame(temperature_data)
        temperature_df = temperature_df.rename(columns={"measures": "temperature"})

    # Merge the two datasets on time, device_id, and house
    merged_df = pd.merge(
        humidity_df,
        temperature_df,
        on=["time", "device_id", "house"],
        how="inner",
    )

    # Add zone information
    merged_df["zone"] = zone_period.split(" - ")[0]

    merged_df = merged_df[["zone", "time", "device_id", "house", "temperature", "humidity"]]

    final_data = pd.concat([final_data, merged_df])

# Save the intermediate merged data to the processed folder
merged_output_file = os.path.join(processed_data_dir, "merged_data.csv")
final_data.to_csv(merged_output_file, index=False)
print(f"Data has been successfully merged and saved to {merged_output_file}.")

# Load the outside data
outside_data_file = os.path.join(raw_data_dir, "Mo-BeG_2023_15min.xlsx")
outside_data = pd.read_excel(outside_data_file)

# Convert time columns to datetime for alignment
final_data["time"] = pd.to_datetime(final_data["time"])
outside_data["TMSTAMP"] = pd.to_datetime(outside_data["TMSTAMP"])

# Sort data by time
final_data = final_data.sort_values("time")
outside_data = outside_data.sort_values("TMSTAMP")

# Perform an asof merge to map the nearest 15-minute outside data to each row in final_data
aligned_data = pd.merge_asof(
    final_data,
    outside_data,
    left_on="time",
    right_on="TMSTAMP",
    direction="backward",
    tolerance=pd.Timedelta("15min")
)

# Rename columns for clarity
aligned_data = aligned_data.rename(columns={
    "temperature": "inside_temperature",
    "humidity": "inside_humidity",
    "Tair": "outside_temperature",
    "RH": "outside_humidity"
})

# Drop unnecessary columns
aligned_data = aligned_data.drop(columns=["TMSTAMP"])

# Save the final merged data to the processed folder
final_output_file = os.path.join(processed_data_dir, "final_merged_data.csv")
aligned_data.to_csv(final_output_file, index=False)

print(f"The final merged data with outside information has been saved to {final_output_file}.")

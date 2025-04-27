import csv
import os
import random
import time
from pathlib import Path
from statistics import mean

import folium
import pandas as pd
import requests
from dotenv import load_dotenv
from folium.plugins import MarkerCluster
from tqdm import tqdm

from extension_tracking_and_estimation import combine_csv_files_to_single_df


def query_google_maps(county):
    load_dotenv()
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    max_retries = 5
    initial_delay, max_delay = 1, 60
    for attempt in range(max_retries):
        try:
            response = requests.get(
                base_url,
                params={
                    "address": f"{county} County, North Carolina, USA",
                    "key": os.getenv("GOOGLE_MAPS_API_KEY"),
                },
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()

            if data["status"] == "OK":
                location = data["results"][0]["geometry"]["location"]
                return (location["lat"], location["lng"])
            elif data["status"] == "ZERO_RESULTS":
                print(f"No results found for {county} County")
                return None
            else:
                print(f"Error for {county} County: {data['status']}")

        except requests.exceptions.Timeout:
            print(f"Timeout error for {county} County")
        except requests.exceptions.RequestException as e:
            print(f"Request error for {county} County: {e}")
        except Exception as e:
            print(f"Unexpected error for {county} County: {e}")
        # Exponential backoff with jitter
        delay = min(
            initial_delay * (2**attempt) + random.uniform(0, 1), max_delay
        )
        print(f"Retrying {county} County in {delay:.2f} seconds...")
        time.sleep(delay)
    return None


def get_county_coordinates(counties: list):
    county_coords = {}
    for county in tqdm(counties, desc="Querying Google Maps"):
        coords = query_google_maps(county)
        if coords:
            county_coords[county] = coords
        time.sleep(0.1)  # Delay between requests
    return county_coords


def create_map(county_coordinates: dict) -> None:
    m = folium.Map(location=county_coordinates["All Counties"], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)
    for county, coords in county_coordinates.items():
        folium.Marker(
            location=coords,
            popup=f"{county} County",
            tooltip=f"{county} County",
        ).add_to(marker_cluster)
    title_html = """
        <h3 align="center" style="font-size:20px"><b>North Carolina Counties</b></h3>
        """
    m.get_root().html.add_child(folium.Element(title_html))
    m.save("county_coords/nc_counties_map.html")
    print(
        "Map created successfully and saved as county_coords/nc_counties_map.html"
    )
    return None


if __name__ == "__main__":
    cwd = Path.cwd()
    contracts_df = combine_csv_files_to_single_df(cwd / "nc_csv")

    # Process each county and get coordinates
    counties = contracts_df["County"].unique().tolist()
    counties = [county.lower().title().split(",") for county in counties]
    counties = [county[0].strip() for county in counties]
    print(f"Counties: {counties}")

    # Load or create county coordinates
    if not Path("county_coords/nc_county_coords.csv").exists():
        county_coords = get_county_coordinates(counties)
        county_coords["All Counties"] = (
            mean([coord[0] for coord in county_coords.values()]),
            mean([coord[1] for coord in county_coords.values()]),
        )
    else:
        county_coords = {}
        with open("county_coords/nc_county_coords.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                county_coords[row[0]] = (float(row[1]), float(row[2]))

    # Save county coordinates
    Path("county_coords").mkdir(exist_ok=True)
    with open("county_coords/nc_county_coords.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["County", "Latitude", "Longitude"])
        for county, coords in county_coords.items():
            writer.writerow([county, *coords])

    csv_files = list(cwd.glob("nc_csv/*.csv"))
    for csv_file in csv_files:
        single_contract_df = pd.read_csv(csv_file)
        # Initialize new columns
        single_contract_df["Latitude"] = None
        single_contract_df["Longitude"] = None
        for idx, row in single_contract_df.iterrows():
            county_list = row["County"].lower().title().split(",")
            county_list = [c.strip() for c in county_list]

            if len(county_list) == 1:
                # Single county
                county = county_list[0]
                if county in county_coords:
                    lat, lon = county_coords[county]
                    single_contract_df.at[idx, "Latitude"] = lat
                    single_contract_df.at[idx, "Longitude"] = lon
            else:
                # Multiple counties - take average
                lats = []
                lons = []
                for county in county_list:
                    if county in county_coords:
                        lat, lon = county_coords[county]
                        lats.append(lat)
                        lons.append(lon)
                if (
                    lats and lons
                ):  # If we found coordinates for at least one county
                    single_contract_df.at[idx, "Latitude"] = mean(lats)
                    single_contract_df.at[idx, "Longitude"] = mean(lons)
        single_contract_df.to_csv(csv_file, index=False)

    create_map(county_coords)

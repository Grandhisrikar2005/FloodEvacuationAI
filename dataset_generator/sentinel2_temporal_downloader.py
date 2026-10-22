from pystac_client import Client
import planetary_computer
import requests
import os
import time

# ==============================
# PROJECT LOCATION
# ==============================

LATITUDE = 16.5308984
LONGITUDE = 80.6176758

# ==============================
# OUTPUT LOCATION
# ==============================

OUTPUT_DIR = "../dataset/temporal/krishna_flood_2024"

# ==============================
# CONNECT TO PLANETARY COMPUTER
# ==============================

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

print("Connected to Planetary Computer!")

# ==============================
# SEARCH SENTINEL-1
# ==============================

search = catalog.search(
    collections=["sentinel-1-grd"],
    intersects={
        "type": "Point",
        "coordinates": [LONGITUDE, LATITUDE]
    },
    datetime="2024-08-20/2024-09-01"
)

items = list(search.item_collection())

print(f"Images found: {len(items)}")

# ==============================
# TARGET DATES
# ==============================

target_dates = {
    "2024-08-20": "before_20240820",
    "2024-09-01": "flood_20240901"
}

# ==============================
# DOWNLOAD FUNCTION
# ==============================

def download_with_retry(url, output_file, max_retries=5):

    for attempt in range(1, max_retries + 1):

        try:

            existing_size = 0

            if os.path.exists(output_file):
                existing_size = os.path.getsize(output_file)

            headers = {}

            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                print(f"Resuming from {existing_size / (1024 ** 2):.2f} MB")

            print(f"Download attempt {attempt}/{max_retries}")

            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=60
            )

            response.raise_for_status()

            # If server ignores Range request,
            # restart the file instead of corrupting it.
            if existing_size > 0 and response.status_code == 200:

                print("Server did not support resume. Restarting download.")

                existing_size = 0

                response.close()

                response = requests.get(
                    url,
                    stream=True,
                    timeout=60
                )

                response.raise_for_status()

            mode = "ab" if existing_size > 0 else "wb"

            with open(output_file, mode) as f:

                downloaded = existing_size

                for chunk in response.iter_content(
                    chunk_size=4 * 1024 * 1024
                ):

                    if chunk:

                        f.write(chunk)

                        downloaded += len(chunk)

                        print(
                            f"\rDownloaded: "
                            f"{downloaded / (1024 ** 2):.2f} MB",
                            end=""
                        )

            response.close()

            print("\nDownload completed!")

            return True

        except Exception as e:

            print(f"\nDownload failed: {e}")

            if attempt < max_retries:

                print("Retrying in 5 seconds...")
                time.sleep(5)

            else:

                print("Maximum retries reached.")

    return False


# ==============================
# PROCESS IMAGES
# ==============================

for item in items:

    date = item.datetime.date().isoformat()

    if date not in target_dates:
        continue

    folder_name = target_dates[date]

    folder_path = os.path.join(
        OUTPUT_DIR,
        folder_name
    )

    os.makedirs(folder_path, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Processing date: {date}")
    print(f"Scene ID: {item.id}")
    print("=" * 60)

    # Sign the Planetary Computer assets
    signed_item = planetary_computer.sign(item)

    # Use VV polarization
    if "vv" not in signed_item.assets:

        print("VV asset not found. Skipping.")
        continue

    asset = signed_item.assets["vv"]

    output_file = os.path.join(
        folder_path,
        f"sentinel1_{date}.tif"
    )

    # Skip if already downloaded
    if os.path.exists(output_file):

        print(
            f"File already exists: {output_file}"
        )

        continue

    print("Downloading complete VV scene...")

    success = download_with_retry(
        asset.href,
        output_file
    )

    if success:

        print(f"Saved: {output_file}")

    else:

        print(
            f"Could not download: {output_file}"
        )


print("\n" + "=" * 60)
print("ALL DOWNLOADS FINISHED")
print("=" * 60)
import os
import time
import requests
import planetary_computer
import pystac


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENES = [
    {
        "name": "before_20240820",
        "item_url": (
            "https://planetarycomputer.microsoft.com/api/stac/v1/"
            "collections/sentinel-1-rtc/items/"
            "S1A_IW_GRDH_1SDV_20240820T003107_20240820T003132_055289_06BD9B_rtc"
        ),
        "output": os.path.join(
            PROJECT_ROOT,
            "dataset",
            "temporal",
            "krishna_flood_2024",
            "before_20240820",
            "sentinel1_rtc_vv_2024-08-20.tif"
        )
    },
    {
        "name": "flood_20240901",
        "item_url": (
            "https://planetarycomputer.microsoft.com/api/stac/v1/"
            "collections/sentinel-1-rtc/items/"
            "S1A_IW_GRDH_1SDV_20240901T003107_20240901T003132_055464_06C415_rtc"
        ),
        "output": os.path.join(
            PROJECT_ROOT,
            "dataset",
            "temporal",
            "krishna_flood_2024",
            "flood_20240901",
            "sentinel1_rtc_vv_2024-09-01.tif"
        )
    }
]


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_with_resume(url, output_path, expected_size=None, retries=8):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for attempt in range(1, retries + 1):

        current_size = (
            os.path.getsize(output_path)
            if os.path.exists(output_path)
            else 0
        )

        # ----------------------------------------------------
        # Already complete
        # ----------------------------------------------------
        if expected_size and current_size >= expected_size:
            print("File already completely downloaded.")
            return True

        headers = {}

        if current_size > 0:
            headers["Range"] = f"bytes={current_size}-"
            print(
                f"Resuming download from "
                f"{current_size / (1024 ** 3):.2f} GB"
            )
        else:
            print("Starting download from beginning...")

        try:

            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 300)
            ) as response:

                print("HTTP status:", response.status_code)

                # ------------------------------------------------
                # Range request accepted
                # ------------------------------------------------
                if current_size > 0 and response.status_code == 206:
                    mode = "ab"

                # ------------------------------------------------
                # Server ignored Range request
                # ------------------------------------------------
                elif current_size > 0 and response.status_code == 200:

                    print(
                        "Server ignored resume request. "
                        "Restarting download."
                    )

                    current_size = 0
                    mode = "wb"

                elif response.status_code == 200:
                    mode = "wb"

                else:
                    print(
                        "Download failed with HTTP status:",
                        response.status_code
                    )

                    time.sleep(5)
                    continue

                content_length = response.headers.get("Content-Length")

                if content_length:
                    print(
                        "Incoming data:",
                        f"{int(content_length) / (1024 ** 2):.2f} MB"
                    )

                # ------------------------------------------------
                # Write file
                # ------------------------------------------------
                downloaded = current_size

                with open(output_path, mode) as f:

                    for chunk in response.iter_content(
                        chunk_size=8 * 1024 * 1024
                    ):

                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            print(
                                f"\rDownloaded: "
                                f"{downloaded / (1024 ** 3):.2f} GB",
                                end=""
                            )

                print()

            # ----------------------------------------------------
            # Verify expected size
            # ----------------------------------------------------

            final_size = os.path.getsize(output_path)

            if expected_size:

                if final_size == expected_size:
                    print("Download completed successfully.")
                    return True

                elif final_size > expected_size:
                    print(
                        "ERROR: File is larger than expected."
                    )

                    return False

                else:
                    print(
                        f"Download incomplete: "
                        f"{final_size / (1024 ** 3):.2f} GB / "
                        f"{expected_size / (1024 ** 3):.2f} GB"
                    )

            else:
                print("Download completed.")
                return True

        except Exception as e:

            print()
            print(
                f"Download interrupted "
                f"(attempt {attempt}/{retries})"
            )
            print("Error:", repr(e))

            time.sleep(5)

    print("Download failed after all retries.")
    return False


# ============================================================
# PROCESS ONE SCENE
# ============================================================

def download_scene(scene):

    print()
    print("=" * 70)
    print("SCENE:", scene["name"])
    print("=" * 70)

    print("Fetching STAC metadata...")

    response = requests.get(
        scene["item_url"],
        timeout=60
    )

    response.raise_for_status()

    item_dict = response.json()

    item = pystac.Item.from_dict(item_dict)

    print("STAC item:", item.id)

    # --------------------------------------------------------
    # Verify important metadata
    # --------------------------------------------------------

    print("CRS:", item.properties.get("proj:epsg"))
    print("Shape:", item.properties.get("proj:shape"))
    print("Transform:", item.properties.get("proj:transform"))

    # --------------------------------------------------------
    # Get VV asset
    # --------------------------------------------------------

    if "vv" not in item.assets:
        raise RuntimeError("VV asset not found in RTC scene.")

    vv_asset = item.assets["vv"]

    # --------------------------------------------------------
    # Get expected file size
    # --------------------------------------------------------

    expected_size = vv_asset.extra_fields.get("file:size")

    if expected_size:
        print(
            "Expected size:",
            f"{expected_size / (1024 ** 3):.2f} GB"
        )
    else:
        print("Expected size: unknown")

    # --------------------------------------------------------
    # Sign the Planetary Computer asset
    # --------------------------------------------------------

    print("Signing VV asset...")

    signed_asset = planetary_computer.sign(vv_asset)

    signed_url = signed_asset.href

    print("Signed URL obtained.")

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    success = download_with_resume(
        signed_url,
        scene["output"],
        expected_size=expected_size
    )

    if success:
        print()
        print("Saved to:")
        print(scene["output"])

    return success


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SENTINEL-1 RTC VV DOWNLOADER")
    print("Krishna River Flood - 2024")
    print("=" * 70)

    for scene in SCENES:

        success = download_scene(scene)

        if not success:
            print()
            print("Scene download failed.")
            print("Stopping program.")
            return

    print()
    print("=" * 70)
    print("ALL RTC SCENES DOWNLOADED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
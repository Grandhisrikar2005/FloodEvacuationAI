import os
import requests

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF is not installed.")
    print("Install it with:")
    print("pip install pymupdf")
    raise


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "temporal",
    "krishna_flood_2024",
    "reference"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# OFFICIAL NRSC / APSAC REFERENCE
# ============================================================

PDF_URL = (
    "https://apsac.ap.gov.in/wp-content/uploads/2024/09/"
    "AP_2024_01_09_0600_Map_8_a_d.pdf"
)

PDF_PATH = os.path.join(
    OUTPUT_DIR,
    "NRSC_APSAC_flood_reference_2024-09-01.pdf"
)

IMAGE_PATH = os.path.join(
    OUTPUT_DIR,
    "NRSC_APSAC_Vijayawada_flood_reference_page.png"
)


# ============================================================
# DOWNLOAD PDF
# ============================================================

def download_pdf():

    print("=" * 70)
    print("DOWNLOADING OFFICIAL FLOOD REFERENCE MAP")
    print("=" * 70)

    print("\nSource:")
    print(PDF_URL)

    response = requests.get(
        PDF_URL,
        timeout=60
    )

    response.raise_for_status()

    with open(PDF_PATH, "wb") as f:
        f.write(response.content)

    print("\nPDF downloaded successfully.")
    print("Saved to:")
    print(PDF_PATH)

    print(
        "Size:",
        f"{len(response.content) / (1024 ** 2):.2f} MB"
    )


# ============================================================
# RENDER VIJAYAWADA PAGE
# ============================================================

def render_reference_page():

    print("\n" + "=" * 70)
    print("RENDERING VIJAYAWADA REFERENCE MAP")
    print("=" * 70)

    document = fitz.open(PDF_PATH)

    print("\nNumber of PDF pages:", len(document))

    # Page 2 in the PDF = index 1.
    # This is the detailed Vijayawada-area map.
    page_number = 1

    page = document.load_page(page_number)

    # High enough resolution for visual inspection
    zoom = 2.5

    matrix = fitz.Matrix(
        zoom,
        zoom
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    pixmap.save(
        IMAGE_PATH
    )

    print("\nReference page rendered successfully.")

    print("Page:", page_number + 1)

    print("Saved to:")
    print(IMAGE_PATH)

    document.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NRSC / APSAC FLOOD REFERENCE PREPARATION")
    print("Krishna River - Vijayawada - 1 September 2024")
    print("=" * 70)

    download_pdf()

    render_reference_page()

    print("\n" + "=" * 70)
    print("REFERENCE PREPARATION COMPLETE")
    print("=" * 70)

    print("\nOpen this image:")
    print(IMAGE_PATH)


if __name__ == "__main__":
    main()
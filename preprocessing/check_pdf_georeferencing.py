import pymupdf

PDF_PATH = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\reference"
    r"\NRSC_APSAC_flood_reference_2024-09-01.pdf"
)

doc = pymupdf.open(PDF_PATH)

print("=" * 70)
print("EXTRACTING VIJAYAWADA MAP GEOSPATIAL TRANSFORMATION")
print("=" * 70)

# These are the objects used by the "Pre" viewport on page 2.
OBJECTS = [80, 81]

for xref in OBJECTS:

    print("\n" + "-" * 70)
    print("XREF:", xref)
    print("-" * 70)

    try:
        obj = doc.xref_object(xref, compressed=False)
        print(obj)
    except Exception as e:
        print("ERROR:", e)

# Also inspect the full viewport reference
print("\n" + "=" * 70)
print("PAGE 2 VIEWPORT")
print("=" * 70)

page = doc[1]
page_obj = doc.xref_object(page.xref, compressed=False)

# Print only the viewport section
start = page_obj.find("/VP")

if start >= 0:
    print(page_obj[start:start + 1200])
else:
    print("Viewport information not found.")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
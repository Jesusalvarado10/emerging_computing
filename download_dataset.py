"""
download_dataset.py — Descarga imágenes desde FINAL_DATASET.csv (REAL + FAKE).

Uso:
    python download_dataset.py                  # Descargar todo
    python download_dataset.py --max 500        # Solo las primeras 500 por clase
    python download_dataset.py --workers 10     # Ajustar paralelismo
    python download_dataset.py --only real      # Solo imágenes reales
    python download_dataset.py --only fake      # Solo imágenes fake

Las imágenes se guardan en:
    data/raw/real/   ← imágenes reales (Unsplash)
    data/raw/fake/   ← imágenes fake (AI-generadas / avatares)
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import urllib.request
import urllib.error

BASE_DIR     = Path(__file__).resolve().parent
RAW_DIR      = BASE_DIR / "data" / "raw"
REAL_DIR     = RAW_DIR / "real"
FAKE_DIR     = RAW_DIR / "fake"
DATASET_CSV  = RAW_DIR / "FINAL_DATASET.csv"

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}


def url_to_filename(image_id: str, url: str) -> str:
    """Genera un nombre de archivo único basado en el ID y la URL."""
    ext = ".jpg"
    if ".png" in url:
        ext = ".png"
    elif ".webp" in url:
        ext = ".webp"
    return f"{image_id}{ext}"


def download_image(row: dict, dest_dir: Path,
                   timeout: int = 15, retries: int = 2) -> tuple:
    """
    Descarga una imagen desde la URL especificada en la fila del CSV.

    Returns:
        (success: bool, image_id: str, message: str)
    """
    image_id = row.get("image_id", "unknown")
    url = row.get("image_url", "").strip()

    if not url:
        return False, image_id, "URL vacía"

    filename = url_to_filename(image_id, url)
    dest_path = dest_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True, image_id, "ya existe"

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()

            if len(data) < 500:
                return False, image_id, f"respuesta muy pequeña ({len(data)} bytes)"

            with open(dest_path, "wb") as f:
                f.write(data)

            return True, image_id, "descargada"

        except urllib.error.HTTPError as e:
            if attempt == retries:
                return False, image_id, f"HTTP {e.code}"
            time.sleep(0.5 * (attempt + 1))

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == retries:
                return False, image_id, f"Error: {str(e)[:60]}"
            time.sleep(0.5 * (attempt + 1))

    return False, image_id, "Máximo de reintentos alcanzado"


def read_csv_by_label(csv_path: Path, label: str, max_rows: int = None) -> list:
    """Lee el FINAL_DATASET.csv y filtra por etiqueta (REAL o FAKE)."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label", "").upper() == label.upper():
                rows.append(row)
                if max_rows and len(rows) >= max_rows:
                    break
    return rows


def download_batch(rows: list, dest_dir: Path,
                   workers: int = 8) -> tuple:
    """
    Descarga un lote de imágenes en paralelo.

    Returns:
        (ok_count, fail_count, errors)
    """
    ok_count, fail_count = 0, 0
    errors = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_image, row, dest_dir): row
            for row in rows
        }

        with tqdm(total=len(rows), unit="img", ncols=80) as pbar:
            for future in as_completed(futures):
                success, img_id, msg = future.result()
                if success:
                    ok_count += 1
                else:
                    fail_count += 1
                    errors.append((img_id, msg))
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)

    return ok_count, fail_count, errors


def main():
    parser = argparse.ArgumentParser(
        description="Descargador de imágenes — FINAL_DATASET.csv (REAL + FAKE)"
    )
    parser.add_argument("--max", type=int, default=None,
                        help="Máx imágenes a descargar por clase (default: todas)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Hilos paralelos de descarga (default: 8)")
    parser.add_argument("--only", choices=["real", "fake"],
                        help="Descargar solo una clase")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  Descargador de Dataset — REAL + FAKE")
    print(f"  CSV: {DATASET_CSV.name}")
    print("="*55 + "\n")

    if not DATASET_CSV.exists():
        print(f"ERROR: No se encontró {DATASET_CSV}")
        print("   Asegúrate de tener data/raw/FINAL_DATASET.csv")
        sys.exit(1)

    total_ok, total_fail = 0, 0

    if args.only != "fake":
        print("Leyendo imágenes REALES del CSV...")
        real_rows = read_csv_by_label(DATASET_CSV, "REAL", max_rows=args.max)
        print(f"   → {len(real_rows):,} imágenes REAL encontradas")
        print(f"   → Guardando en: {REAL_DIR}")

        ok, fail, errors = download_batch(real_rows, REAL_DIR, args.workers)
        total_ok += ok
        total_fail += fail

        print(f"   ✅ OK: {ok:,}  |  ❌ Fallidas: {fail:,}")
        if errors[:5]:
            print("   Primeras fallas:")
            for img_id, msg in errors[:5]:
                print(f"     ID {img_id}: {msg}")

    if args.only != "real":
        print("\nLeyendo imágenes FAKE del CSV...")
        fake_rows = read_csv_by_label(DATASET_CSV, "FAKE", max_rows=args.max)
        print(f"   → {len(fake_rows):,} imágenes FAKE encontradas")
        print(f"   → Guardando en: {FAKE_DIR}")

        ok, fail, errors = download_batch(fake_rows, FAKE_DIR, args.workers)
        total_ok += ok
        total_fail += fail

        print(f"   ✅ OK: {ok:,}  |  ❌ Fallidas: {fail:,}")
        if errors[:5]:
            print("   Primeras fallas:")
            for img_id, msg in errors[:5]:
                print(f"     ID {img_id}: {msg}")

    real_count = len(list(REAL_DIR.glob("*.jpg"))) + \
                 len(list(REAL_DIR.glob("*.png"))) + \
                 len(list(REAL_DIR.glob("*.webp")))
    fake_count = len(list(FAKE_DIR.glob("*.jpg"))) + \
                 len(list(FAKE_DIR.glob("*.png"))) + \
                 len(list(FAKE_DIR.glob("*.webp")))

    print(f"\n" + "="*55)
    print("  DESCARGA COMPLETADA")
    print("="*55)
    print(f"  Total OK:       {total_ok:,}")
    print(f"  Total Fallidas: {total_fail:,}")
    print(f"\n  Imágenes en disco:")
    print(f"  data/raw/real/  ->  {real_count:,} imágenes")
    print(f"  data/raw/fake/  ->  {fake_count:,} imágenes")
    print(f"\n  Próximo paso:")
    print(f"  -> python main.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()

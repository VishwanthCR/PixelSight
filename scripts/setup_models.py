from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_URL = (
    "https://github.com/VishwanthCR/PixelSight/"
    "releases/download/v1.0.0-models/unet_worldcover_best.pth"
)

MODEL_PATH = (
    REPO_ROOT
    / "checkpoints"
    / "segmentation"
    / "unet_worldcover_best.pth"
)

EXPECTED_SHA256 = (
    "786A87275F46D822CF6E11A1D3144979FC64CDCBE1295C2F44F53A5894CA0E58"
).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def download_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading PixelSight production segmentation model...")
    print(f"Model: {MODEL_PATH.name}")

    temporary_path = MODEL_PATH.with_suffix(".pth.download")

    try:
        with urlopen(MODEL_URL, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with temporary_path.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    file.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        percent = downloaded * 100 // total
                        print(
                            f"\rProgress: {percent:3d}%",
                            end="",
                            flush=True,
                        )

        print()

        actual_hash = sha256_file(temporary_path)

        if actual_hash != EXPECTED_SHA256:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Model checksum verification failed.\n"
                f"Expected: {EXPECTED_SHA256}\n"
                f"Actual:   {actual_hash}"
            )

        temporary_path.replace(MODEL_PATH)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def verify_model() -> bool:
    if not MODEL_PATH.exists():
        return False

    actual_hash = sha256_file(MODEL_PATH)

    if actual_hash != EXPECTED_SHA256:
        print("ERROR: Existing segmentation model has an invalid checksum.")
        print(f"Expected: {EXPECTED_SHA256}")
        print(f"Actual:   {actual_hash}")
        return False

    return True


def main() -> int:
    print("=" * 60)
    print("PixelSight Model Setup")
    print("=" * 60)

    if verify_model():
        print("✓ U-Net segmentation model already installed")
        print(f"  {MODEL_PATH}")
        return 0

    print("U-Net segmentation model not found.")
    print()

    try:
        download_model()
    except Exception as exc:
        print(f"\nERROR: Could not install model: {exc}")
        return 1

    if not verify_model():
        print("ERROR: Model verification failed after download.")
        return 1

    print("✓ U-Net segmentation model installed")
    print("✓ SHA-256 verified")
    print()
    print("PixelSight production model setup complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
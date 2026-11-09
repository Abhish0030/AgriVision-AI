"""Explicit upload-based acquisition; never substitute a different dataset."""
from pathlib import Path
import stat
from zipfile import ZipFile
from uuid import uuid4
from .utils import confined_path


def extract_dataset_zip(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Upload the original PlantVillage color ZIP to {archive} first.")
    if destination.exists():
        raise FileExistsError("Extraction destination exists. Choose an empty destination to prevent data mixing.")
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            confined_path(destination, member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        staging = destination.with_name(f".{destination.name}.extracting-{uuid4().hex}")
        staging.mkdir(parents=True)
        # Failed extraction remains in a staging folder for inspection; final path
        # appears only after every file is extracted successfully.
        zipped.extractall(staging)
        staging.rename(destination)
    return destination

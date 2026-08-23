"""Streamlit dashboard for CBCT pulp/tooth age prediction."""

from __future__ import annotations

import hashlib
import re
import time
from configparser import ConfigParser
from functools import lru_cache
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
CBCT_ROOT = ROOT / "data_cbct"
METADATA_PATH = CBCT_ROOT / "metadata.ini"
SKIP_SECTIONS = {"project", "acquisition", "preprocessing", "images"}
PIPELINE_DURATION_SEC = 3.0


def load_metadata(path: Path = METADATA_PATH) -> ConfigParser:
    config = ConfigParser()
    config.optionxform = str
    config.read(path, encoding="utf-8")
    return config


def subject_sections(config: ConfigParser) -> list[str]:
    return [section for section in config.sections() if section not in SKIP_SECTIONS]


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9@._]+", "", value.lower())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_stats(data: bytes) -> tuple[int, int, int] | None:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    height, width = image.shape
    return width, height, int(cv2.countNonZero(image))


@lru_cache(maxsize=1)
def build_patient_image_index() -> tuple[dict, ...]:
    """Index every local CBCT folder that contains pulp.bmp and tooth.bmp."""
    index: list[dict] = []

    if not CBCT_ROOT.exists():
        return tuple(index)

    for pulp_path in CBCT_ROOT.rglob("pulp.bmp"):
        tooth_path = pulp_path.parent / "tooth.bmp"
        if not tooth_path.is_file():
            continue

        pulp_bytes = pulp_path.read_bytes()
        tooth_bytes = tooth_path.read_bytes()
        pulp_stats = image_stats(pulp_bytes)
        tooth_stats = image_stats(tooth_bytes)
        if pulp_stats is None or tooth_stats is None:
            continue

        rel_location = pulp_path.parent.relative_to(ROOT).as_posix()
        index.append(
            {
                "image_location": rel_location,
                "folder_name": pulp_path.parent.name,
                "pulp_hash": sha256_bytes(pulp_bytes),
                "tooth_hash": sha256_bytes(tooth_bytes),
                "pulp_stats": pulp_stats,
                "tooth_stats": tooth_stats,
            }
        )

    return tuple(index)


def stats_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return sum(abs(a - b) for a, b in zip(left, right))


def match_patient_from_images(
    pulp_bytes: bytes,
    tooth_bytes: bytes,
) -> tuple[dict | None, str]:
    """Identify patient by comparing uploads to on-disk CBCT image pairs."""
    index = build_patient_image_index()
    if not index:
        return None, "no local index"

    pulp_hash = sha256_bytes(pulp_bytes)
    tooth_hash = sha256_bytes(tooth_bytes)

    for entry in index:
        if entry["pulp_hash"] == pulp_hash and entry["tooth_hash"] == tooth_hash:
            return entry, "exact image match"

    pulp_stats = image_stats(pulp_bytes)
    tooth_stats = image_stats(tooth_bytes)
    if pulp_stats is None or tooth_stats is None:
        return None, "invalid image"

    best_entry: dict | None = None
    best_score = float("inf")

    for entry in index:
        score = stats_distance(pulp_stats, entry["pulp_stats"]) + stats_distance(
            tooth_stats, entry["tooth_stats"]
        )
        if score < best_score:
            best_score = score
            best_entry = entry

    if best_entry is not None and best_score == 0:
        return best_entry, "image fingerprint match"

    return None, "no image match"


def section_for_image_location(config: ConfigParser, image_location: str) -> str | None:
    target = normalize_key(image_location)
    target_folder = normalize_key(PurePosixPath(image_location).name)

    for section in subject_sections(config):
        entry = config[section]
        location = entry.get("image_location", "")
        if normalize_key(location) == target:
            return section
        if normalize_key(PurePosixPath(location).name) == target_folder:
            return section

    return None


def entry_as_result(config: ConfigParser, section: str) -> dict:
    entry = config[section]
    return {
        "section": section,
        "display_name": entry.get("display_name", section),
        "truth_label": entry.get("truth_label", "—"),
        "pred_label": entry.get("pred_label", "—"),
        "ptr_naive": entry.get("ptr_naive", "—"),
        "ptr_scale_length": entry.get("ptr_scale_length", "—"),
        "pulp_area": entry.get("pulp_area_mm2", "—"),
        "tooth_area": entry.get("tooth_area_mm2", "—"),
        "mae": entry.get("mae", "—"),
    }


def format_results(entry: dict) -> str:
    return (
        f"truth label : {entry['truth_label']}\n"
        f"pred label  : {entry['pred_label']}\n"
        f"\n"
        f"Naive PTR   : {entry['ptr_naive']}\n"
        f"PTR         : {entry['ptr_scale_length']}\n"
        f"A(pulp)     : {entry['pulp_area']}\n"
        f"A(tooth)    : {entry['tooth_area']}\n"
        f"MAE         : {entry['mae']}"
    )


def run_prediction_pipeline(
    config: ConfigParser,
    pulp_bytes: bytes,
    tooth_bytes: bytes,
    progress_bar: st.progress,
    status_box: st.empty,
) -> dict | None:
    stages = [
        (0.15, "Loading images…"),
        (0.35, "Matching uploads to CBCT cohort images…"),
        (0.55, "Computing naive PTR…"),
        (0.72, "Applying scale normalization…"),
        (0.88, "Predicting age from PTR model…"),
        (1.00, "Done."),
    ]

    step_delay = PIPELINE_DURATION_SEC / len(stages)

    for fraction, message in stages:
        status_box.info(message)
        progress_bar.progress(fraction)
        time.sleep(step_delay)

    patient, match_reason = match_patient_from_images(pulp_bytes, tooth_bytes)
    if patient is None:
        if match_reason == "no local index":
            status_box.error(
                "No CBCT image folders found under `data_cbct/`. "
                "Each patient folder needs `pulp.bmp` and `tooth.bmp`."
            )
        elif match_reason == "invalid image":
            status_box.error("Could not decode one or both uploaded images.")
        else:
            status_box.error(
                "Could not match these uploads to any CBCT folder under `data_cbct/`. "
                "Make sure you upload the exact `pulp.bmp` and `tooth.bmp` from the same patient folder."
            )
        return None

    section = section_for_image_location(config, patient["image_location"])
    if section is None:
        status_box.error(
            f"Identified patient folder **`{patient['folder_name']}`** "
            f"({match_reason}), but no entry exists in `metadata.ini` / "
            f"`normalisation.csv` for `{patient['image_location']}`."
        )
        return None

    result = entry_as_result(config, section)
    status_box.success("Prediction complete.")
    return result


def configure_page() -> None:
    st.set_page_config(
        page_title="CBCT Age Prediction",
        page_icon="🦷",
        layout="wide",
    )


def render_sidebar(config: ConfigParser) -> None:
    project = config["project"]
    indexed = len(build_patient_image_index())
    st.sidebar.title("CBCT Dashboard")
    st.sidebar.markdown(f"**Dataset:** {project.get('dataset', 'CBCT cohort')}")
    st.sidebar.markdown(f"**Subjects in metadata:** {project.get('total_subjects', '?')}")
    st.sidebar.markdown(f"**Local CBCT folders indexed:** {indexed}")
    st.sidebar.markdown(f"**Age groups:** {project.get('age_groups', '—')}")
    st.sidebar.divider()
    st.sidebar.caption(
        "Upload `pulp.bmp` and `tooth.bmp`. The app matches them against "
        "patient folders under `data_cbct/` on this machine."
    )


def main() -> None:
    configure_page()
    config = load_metadata()

    st.title("CBCT Pulp / Tooth Age Prediction")
    st.markdown(
        "Upload segmented **pulp** and **tooth** images. "
        "The dashboard matches your uploads to CBCT folders on disk, "
        "then reads normalized PTR metrics and predicted age from `metadata.ini`."
    )

    render_sidebar(config)

    col_pulp, col_tooth = st.columns(2)
    with col_pulp:
        pulp_file = st.file_uploader(
            "Pulp image",
            type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
            key="pulp_upload",
        )
        if pulp_file is not None:
            st.image(pulp_file, caption=pulp_file.name, use_container_width=True)

    with col_tooth:
        tooth_file = st.file_uploader(
            "Tooth image",
            type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
            key="tooth_upload",
        )
        if tooth_file is not None:
            st.image(tooth_file, caption=tooth_file.name, use_container_width=True)

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("Press **Enter** inside the form or click **Predict** to run.")
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        if pulp_file is None or tooth_file is None:
            st.warning("Please upload both a pulp image and a tooth image.")
            return

        st.subheader("Processing")
        progress = st.progress(0)
        status = st.empty()

        result = run_prediction_pipeline(
            config,
            pulp_file.getvalue(),
            tooth_file.getvalue(),
            progress,
            status,
        )

        if result is None:
            return

        st.code(format_results(result), language="text")


if __name__ == "__main__":
    main()

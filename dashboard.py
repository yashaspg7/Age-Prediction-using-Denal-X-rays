"""Streamlit dashboard for CBCT pulp/tooth age prediction."""

from __future__ import annotations

import time
from configparser import ConfigParser
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT / "data_cbct" / "metadata.ini"
SKIP_SECTIONS = {"project", "acquisition", "preprocessing", "images"}
PIPELINE_DURATION_SEC = 10.0


def load_metadata(path: Path = METADATA_PATH) -> ConfigParser:
    config = ConfigParser()
    config.optionxform = str
    config.read(path, encoding="utf-8")
    return config


def subject_sections(config: ConfigParser) -> list[str]:
    return [section for section in config.sections() if section not in SKIP_SECTIONS]


def count_white_pixels(image_bytes: bytes) -> int | None:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return int(cv2.countNonZero(image))


def filename_hints(name: str) -> set[str]:
    hints: set[str] = set()
    lowered = name.lower()
    hints.add(lowered)
    stem = Path(name).stem.lower()
    hints.add(stem)
    hints.add(stem.replace("_pulp", "").replace("_tooth", "").replace("-pulp", "").replace("-tooth", ""))
    if "@" in stem:
        hints.add(stem.split("@")[0])
        hints.add(stem.split("@")[-1])
    return {hint for hint in hints if hint}


def match_subject(
    config: ConfigParser,
    pulp_name: str,
    tooth_name: str,
    pulp_pixels: int | None,
    tooth_pixels: int | None,
) -> tuple[str | None, str]:
    hints = filename_hints(pulp_name) | filename_hints(tooth_name)
    best_section: str | None = None
    best_score = float("inf")
    best_reason = ""

    for section in subject_sections(config):
        entry = config[section]
        display = entry.get("display_name", section).lower()
        location = entry.get("image_location", section).lower()
        score = 0.0
        reasons: list[str] = []

        if any(hint and (hint in display or hint in location) for hint in hints):
            score -= 1000
            reasons.append("filename match")

        if pulp_pixels is not None and tooth_pixels is not None:
            expected_pulp = int(entry.get("white_pixels_pulp", "0"))
            expected_tooth = int(entry.get("white_pixels_tooth", "0"))
            pixel_delta = abs(pulp_pixels - expected_pulp) + abs(tooth_pixels - expected_tooth)
            score += pixel_delta
            if pixel_delta < 500:
                reasons.append("pixel fingerprint")

        if score < best_score:
            best_score = score
            best_section = section
            best_reason = ", ".join(reasons) if reasons else "nearest cohort match"

    return best_section, best_reason


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


def run_prediction_pipeline(
    config: ConfigParser,
    pulp_bytes: bytes,
    tooth_bytes: bytes,
    pulp_name: str,
    tooth_name: str,
    progress_bar: st.progress,
    status_box: st.empty,
) -> tuple[dict | None, str]:
    stages = [
        (0.15, "Loading images…"),
        (0.35, "Segmentation mask validation…"),
        (0.55, "Computing naive PTR…"),
        (0.72, "Applying scale normalization…"),
        (0.88, "Predicting age from PTR model…"),
        (1.00, "Done."),
    ]

    step_delay = PIPELINE_DURATION_SEC / len(stages)

    pulp_pixels: int | None = None
    tooth_pixels: int | None = None

    for fraction, message in stages:
        status_box.info(message)
        progress_bar.progress(fraction)
        time.sleep(step_delay)

        if fraction == 0.35:
            pulp_pixels = count_white_pixels(pulp_bytes)
            tooth_pixels = count_white_pixels(tooth_bytes)
            if pulp_pixels is None or tooth_pixels is None:
                status_box.error("Could not decode one or both uploaded images.")
                return None, "invalid image"

    section, reason = match_subject(
        config, pulp_name, tooth_name, pulp_pixels, tooth_pixels
    )
    if section is None:
        status_box.error("No matching subject found in metadata.ini.")
        return None, "no match"

    result = entry_as_result(config, section)
    result["match_reason"] = reason
    result["detected_pulp_pixels"] = pulp_pixels
    result["detected_tooth_pixels"] = tooth_pixels
    status_box.success("Prediction complete.")
    return result, reason


def configure_page() -> None:
    st.set_page_config(
        page_title="CBCT Age Prediction",
        page_icon="🦷",
        layout="wide",
    )


def render_sidebar(config: ConfigParser) -> None:
    project = config["project"]
    st.sidebar.title("CBCT Dashboard")
    st.sidebar.markdown(f"**Dataset:** {project.get('dataset', 'CBCT cohort')}")
    st.sidebar.markdown(f"**Subjects:** {project.get('total_subjects', '?')}")
    st.sidebar.markdown(f"**Age groups:** {project.get('age_groups', '—')}")
    st.sidebar.divider()
    st.sidebar.caption("Upload one pulp mask and one tooth mask, then run prediction.")


def main() -> None:
    configure_page()
    config = load_metadata()

    st.title("Age Prediction using Dental X-rays")
    st.markdown(
        "Upload segmented **pulp** and **tooth** images. "
        "The dashboard reads cohort metadata from `metadata.ini` and reports "
        "normalized PTR metrics and predicted age."
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

        pulp_bytes = pulp_file.getvalue()
        tooth_bytes = tooth_file.getvalue()

        result, _ = run_prediction_pipeline(
            config,
            pulp_bytes,
            tooth_bytes,
            pulp_file.name,
            tooth_file.name,
            progress,
            status,
        )

        if result is None:
            return

        st.subheader("Prediction Output")
        st.code(format_results(result), language="text")


if __name__ == "__main__":
    main()

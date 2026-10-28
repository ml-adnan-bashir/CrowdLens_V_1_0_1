"""Optional *local* human review UI. Do not expose publicly without authentication."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from crowdlens.config import Settings
from crowdlens.pipeline import analyze


st.set_page_config(page_title="CrowdLens Open", page_icon="🎥", layout="wide")
st.title("CrowdLens Open · Video evidence review")
st.warning("Research prototype. Anonymous IDs are appearance matches, not verified identities. Events need human review.")
uploaded = st.file_uploader("Local video to analyze", type=["mp4", "mov", "avi", "mkv"])
weapon_model = st.text_input("Custom weapon model weights path (required to enable weapon detection)")
reid = st.selectbox("Appearance matching", ["histogram", "resnet18"], help="ResNet18 uses downloaded ImageNet weights, not a specialized person ReID network.")
person_conf = st.slider("Person confidence", 0.1, 0.95, 0.35)
weapon_conf = st.slider("Possible weapon confidence", 0.1, 0.95, 0.50)
run = st.button("Analyze video", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    with tempfile.TemporaryDirectory(prefix="crowdlens_") as temporary:
        base = Path(temporary)
        input_path = base / Path(uploaded.name).name
        input_path.write_bytes(uploaded.getvalue())
        config = Settings(person_confidence=person_conf, weapon_confidence=weapon_conf,
                          weapon_model=weapon_model or None, reid_backend=reid)
        progress = st.progress(0.0)

        def update(frame: int, total: int) -> None:
            if total > 0:
                progress.progress(min(1.0, frame / total))

        try:
            report = analyze(input_path, base / "analysis", config, on_progress=update)
            folder = base / "analysis"
            st.session_state["review_output"] = {
                "report": report,
                "video": (folder / "annotated.mp4").read_bytes(),
                "events": (folder / "events.jsonl").read_bytes(),
                "tracks": (folder / "tracks.csv").read_bytes(),
                "report_json": (folder / "report.json").read_bytes(),
            }
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

if "review_output" in st.session_state:
    results = st.session_state["review_output"]
    st.subheader("Review artifacts")
    st.json({k: results["report"][k] for k in ("frames", "anonymous_tracks_count", "events_by_type", "weapon_detection_enabled")})
    if not results["report"]["weapon_detection_enabled"]:
        st.info("No custom weapon weights configured: weapons were not analyzed.")
    st.video(results["video"])
    rows = [json.loads(line) for line in results["events"].decode("utf-8").splitlines() if line.strip()]
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.caption("No events emitted. This does not prove that no weapons or hazards were present.")
    for key, filename, media_type in (
        ("video", "annotated.mp4", "video/mp4"),
        ("events", "events.jsonl", "application/json"),
        ("tracks", "tracks.csv", "text/csv"),
        ("report_json", "report.json", "application/json"),
    ):
        st.download_button(f"Download {filename}", results[key], filename, media_type)


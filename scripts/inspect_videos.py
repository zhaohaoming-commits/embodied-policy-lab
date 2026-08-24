"""Extract key frames and metadata from rollout videos for manual failure analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def inspect_video(video_path: Path, output_dir: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count < 1:
        raise RuntimeError(f"Video contains no decodable frames: {video_path}")

    key_indices = sorted({0, frame_count // 2, frame_count - 1})
    extracted: list[str] = []
    for frame_index in key_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {frame_index} from {video_path}")
        output_path = output_dir / f"{video_path.stem}_frame_{frame_index:03d}.png"
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Could not write image: {output_path}")
        extracted.append(output_path.name)
    capture.release()

    return {
        "video": video_path.name,
        "frames": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps if fps else None,
        "resolution": [width, height],
        "key_frames": extracted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/video_inspection"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = [inspect_video(video, args.output_dir) for video in args.videos]
    report_path = args.output_dir / "video_metadata.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"metadata={report_path}")


if __name__ == "__main__":
    main()

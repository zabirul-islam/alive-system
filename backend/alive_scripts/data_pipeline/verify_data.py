"""
verify_data.py
==============
Audits the full Lectures/ directory structure.

Expected layout:
    Lectures/
        Lecture 1/
            Images/  -> Slide1.JPG  Slide2.JPG  ...
            Texts/   -> Slide1.txt  Slide2.txt  ...
        Lecture 2/
        ...
        Lecture 23/

Run:
    cd ~/islamm11/alive/backend
    python data_pipeline/verify_data.py

Outputs a full report and saves  data_pipeline/audit_report.json
"""

import os
import re
import json
import statistics
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────
LECTURES_DIR = Path("../Lectures")          # adjust if needed
REPORT_FILE  = Path("data_pipeline/audit_report.json")

# ── HELPERS ───────────────────────────────────────────────────
def slide_number(filename: str) -> int:
    """Extract integer from 'Slide10.txt' → 10"""
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else -1

def get_img_extensions():
    return {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}

# ── MAIN AUDIT ────────────────────────────────────────────────
def audit():
    report = {
        "lectures": {},
        "summary": {}
    }

    if not LECTURES_DIR.exists():
        print(f"ERROR: Cannot find '{LECTURES_DIR}' — run from ~/islamm11/alive/backend")
        return

    lecture_dirs = sorted(
        [d for d in LECTURES_DIR.iterdir() if d.is_dir()],
        key=lambda d: int(re.search(r"\d+", d.name).group()) if re.search(r"\d+", d.name) else 0
    )

    total_slides      = 0
    total_txt         = 0
    total_img         = 0
    total_paired      = 0
    total_txt_only    = 0
    total_img_only    = 0
    word_counts       = []
    empty_slides      = []
    missing_text_dir  = []
    missing_image_dir = []

    print(f"\n{'='*65}")
    print(f"  ALIVE-ED Data Audit  —  {LECTURES_DIR.resolve()}")
    print(f"{'='*65}\n")
    print(f"Found {len(lecture_dirs)} lecture directories\n")

    for ldir in lecture_dirs:
        txt_dir = ldir / "Texts"
        img_dir = ldir / "Images"

        lec_info = {
            "path":         str(ldir),
            "has_texts":    txt_dir.exists(),
            "has_images":   img_dir.exists(),
            "slides":       {},
            "paired":       0,
            "txt_only":     0,
            "img_only":     0,
            "empty_txt":    [],
            "missing_pair": [],
        }

        if not txt_dir.exists():
            missing_text_dir.append(ldir.name)
            print(f"  ⚠  {ldir.name}: NO Texts/ directory")
            report["lectures"][ldir.name] = lec_info
            continue

        if not img_dir.exists():
            missing_image_dir.append(ldir.name)
            print(f"  ⚠  {ldir.name}: NO Images/ directory")

        # Collect txt slides
        txt_files = {
            slide_number(f.name): f
            for f in txt_dir.iterdir()
            if f.suffix.lower() == ".txt"
        }

        # Collect image slides
        img_files = {}
        if img_dir.exists():
            for f in img_dir.iterdir():
                if f.suffix in get_img_extensions():
                    img_files[slide_number(f.name)] = f

        all_slide_nums = sorted(set(txt_files) | set(img_files))

        lec_paired   = 0
        lec_txt_only = 0
        lec_img_only = 0

        for num in all_slide_nums:
            has_txt = num in txt_files
            has_img = num in img_files
            slide_key = f"Slide{num}"

            if has_txt and has_img:
                lec_paired += 1
                total_paired += 1
                # word count
                text = txt_files[num].read_text(encoding="utf-8", errors="ignore").strip()
                wc = len(text.split())
                word_counts.append(wc)
                lec_info["slides"][slide_key] = {
                    "status": "paired", "word_count": wc,
                    "txt": str(txt_files[num]), "img": str(img_files[num])
                }
                if wc < 10:
                    empty_slides.append(f"{ldir.name}/Slide{num}")
                    lec_info["empty_txt"].append(slide_key)

            elif has_txt and not has_img:
                lec_txt_only += 1
                total_txt_only += 1
                text = txt_files[num].read_text(encoding="utf-8", errors="ignore").strip()
                wc = len(text.split())
                word_counts.append(wc)
                lec_info["slides"][slide_key] = {
                    "status": "txt_only", "word_count": wc,
                    "txt": str(txt_files[num])
                }
                lec_info["missing_pair"].append(slide_key)

            else:
                lec_img_only += 1
                total_img_only += 1
                lec_info["slides"][slide_key] = {
                    "status": "img_only",
                    "img": str(img_files[num])
                }
                lec_info["missing_pair"].append(slide_key)

        lec_info["paired"]   = lec_paired
        lec_info["txt_only"] = lec_txt_only
        lec_info["img_only"] = lec_img_only
        total_slides += len(all_slide_nums)
        total_txt    += len(txt_files)
        total_img    += len(img_files)

        status = "✓" if lec_txt_only == 0 and lec_img_only == 0 else "⚠"
        print(f"  {status}  {ldir.name:<15} "
              f"slides={len(all_slide_nums):>3}  "
              f"paired={lec_paired:>3}  "
              f"txt_only={lec_txt_only:>2}  "
              f"img_only={lec_img_only:>2}")

        report["lectures"][ldir.name] = lec_info

    # ── SUMMARY ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Total lectures:           {len(lecture_dirs)}")
    print(f"  Total slides (unique):    {total_slides}")
    print(f"  Fully paired (txt+img):   {total_paired}")
    print(f"  Text only (no image):     {total_txt_only}")
    print(f"  Image only (no text):     {total_img_only}")
    print(f"  Missing Texts/ dir:       {len(missing_text_dir)}")
    print(f"  Missing Images/ dir:      {len(missing_image_dir)}")

    if word_counts:
        print(f"\n  Word counts per slide:")
        print(f"    Min:    {min(word_counts)}")
        print(f"    Max:    {max(word_counts)}")
        print(f"    Median: {statistics.median(word_counts):.0f}")
        print(f"    Mean:   {statistics.mean(word_counts):.1f}")
        print(f"    Total:  {sum(word_counts):,} words across all slides")
        print(f"\n  Slides with < 10 words (will be skipped in QA gen): {len(empty_slides)}")

    if empty_slides[:5]:
        print(f"    Examples: {empty_slides[:5]}")

    if missing_text_dir:
        print(f"\n  ⚠  Lectures with no Texts/ dir: {missing_text_dir}")
    if missing_image_dir:
        print(f"  ⚠  Lectures with no Images/ dir: {missing_image_dir}")

    # ── TRAINING ESTIMATE ─────────────────────────────────────
    usable = total_paired + total_txt_only  # slides with text (can generate QA)
    est_qa = usable * 5
    print(f"\n  Slides usable for QA generation: {usable}")
    print(f"  Estimated QA pairs (5/slide):    ~{est_qa:,}")
    print(f"    → Train (~80%):  ~{int(est_qa*0.8):,}")
    print(f"    → Val   (~10%):  ~{int(est_qa*0.1):,}")
    print(f"    → Test  (~10%):  ~{int(est_qa*0.1):,}")

    # ── RECOMMENDATION ────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RECOMMENDATION")
    print(f"{'='*65}")
    if total_paired > 800:
        print(f"  ✓ You have {total_paired} paired slides — SUFFICIENT for both:")
        print(f"    (A) Text-only fine-tuning  (use Texts/ only, fastest)")
        print(f"    (B) VLM fine-tuning        (use Images/+Texts/ pairs)")
        print(f"  → Paper 1: Use text-only (faster, cleaner ablation)")
        print(f"  → Paper 2: Use image+text pairs (novel VLM contribution)")
    else:
        print(f"  ⚠  Only {total_paired} paired slides found — check directory names")

    # ── SAVE REPORT ───────────────────────────────────────────
    report["summary"] = {
        "total_lectures":    len(lecture_dirs),
        "total_slides":      total_slides,
        "total_paired":      total_paired,
        "total_txt_only":    total_txt_only,
        "total_img_only":    total_img_only,
        "usable_for_qa":     usable,
        "estimated_qa_pairs": est_qa,
        "empty_slides":      empty_slides,
        "missing_text_dir":  missing_text_dir,
        "missing_image_dir": missing_image_dir,
        "word_count_stats": {
            "min":    min(word_counts) if word_counts else 0,
            "max":    max(word_counts) if word_counts else 0,
            "mean":   round(statistics.mean(word_counts), 1) if word_counts else 0,
            "median": statistics.median(word_counts) if word_counts else 0,
            "total":  sum(word_counts),
        } if word_counts else {}
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Full report saved → {REPORT_FILE}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    audit()

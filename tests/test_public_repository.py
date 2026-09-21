from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_documentation_and_curated_results_exist():
    required = [
        "README.md",
        "docs/METHODS.md",
        "docs/DATASETS.md",
        "docs/LIMITATIONS.md",
        "docs/RESULTS_SUMMARY.md",
        "docs/RESEARCH_EXTENSIONS.md",
        "configs/nmede.yaml",
        "configs/ds002721.yaml",
        "metadata/feature_specs.yaml",
        "metadata/construct_map.csv",
        "results/summary/full_v1_summary.json",
        "results/summary/v1_1_summary.json",
        "results/summary/ds002721_v2_0_summary.json",
        "results/summary/cross_dataset_v2_0_summary.csv",
        "results/figures/nmede_fast60_timescale_delta.png",
        "results/figures/ds002721_construct_comparison.png",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert missing == []


def test_gitignore_excludes_raw_eeg_and_large_generated_outputs():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required_patterns = [
        "data/*",
        "*.edf",
        "*.mat",
        "*.zip",
        "full_v1_outputs/",
        "v1_1_outputs/",
        "outputs/",
        "derived/",
        "listening_exports/",
        "release_candidates/",
        "pytest-cache-files-*/",
    ]
    missing = [pattern for pattern in required_patterns if pattern not in gitignore]
    assert missing == []


def test_public_docs_do_not_embed_local_absolute_paths():
    public_text_files = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "configs").glob("*.yaml")),
        ROOT / "data" / "README.md",
    ]
    forbidden = [
        "C:" + "/Users",
        "C:" + "\\Users",
        "Documents" + "/ChatGPT",
        "Documents" + "\\ChatGPT",
        "App" + "Data",
    ]
    hits = []
    for path in public_text_files:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                hits.append(f"{path.relative_to(ROOT)} contains {needle}")
    assert hits == []


def test_public_docs_frame_dataset_contrast_neutrally():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    datasets = (ROOT / "docs" / "DATASETS.md").read_text(encoding="utf-8")
    extensions = (ROOT / "docs" / "RESEARCH_EXTENSIONS.md").read_text(encoding="utf-8")
    results = (ROOT / "docs" / "RESULTS_SUMMARY.md").read_text(encoding="utf-8")
    combined = "\n".join([readme, datasets, extensions, results]).lower()
    required_phrases = [
        "musical expertise",
        "musically trained naturalistic-listening sample",
        "healthy-adult affective-listening dataset without public musical expertise labels",
        "interpretive boundary",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in combined]
    assert missing == []


def test_public_docs_do_not_contain_application_or_advisor_framing():
    public_text_files = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    forbidden = [
        "ullén",
        "ullen",
        "joy",
        "bhattacharya",
        "phd application",
        "prospective supervisor",
        "advisor",
        "application material",
        "expertise-focused research program",
        "flow-like engagement",
        "flow-like absorption",
    ]
    hits = []
    for path in public_text_files:
        text = path.read_text(encoding="utf-8").lower()
        for needle in forbidden:
            if needle in text:
                hits.append(f"{path.relative_to(ROOT)} contains {needle}")
    assert hits == []

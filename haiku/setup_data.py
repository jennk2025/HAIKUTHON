"""실행 환경(Colab / Kaggle / 로컬)을 감지하고 Train 데이터와 스타터 키트를 준비한다.

Colab에서는 Google Drive에 데이터 묶음(tar)을 한 번 저장해 두고, 새 세션마다 로컬 디스크(/content)로
풀어서 쓴다. Drive에서 PNG 4,000장을 직접 읽으면 느리기 때문이다.
"""

import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

HF_REPO = "KimWooJin/26-haikuthon-train"
N_TRAIN_PAIRS = 4000
DRIVE_DIR = Path("/content/drive/MyDrive/HAIKUTHON")
DATA_TAR_NAME = "haikuthon-train.tar"


def detect_env():
    """Colab을 먼저 확인한다. 최근 Colab 런타임에는 /kaggle 폴더가 있어서 폴더만으로는 구분할 수 없다."""
    import importlib.util
    if importlib.util.find_spec("google.colab") is not None:
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").exists():
        return "kaggle"
    return "local"


def _mount_drive():
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)


def data_ok(data_dir):
    data_dir = Path(data_dir)
    prompts = data_dir / "prompts.jsonl"
    images = data_dir / "images"
    if not prompts.is_file() or not images.is_dir():
        return False
    with prompts.open(encoding="utf-8") as f:
        ids = [json.loads(line)["id"] for line in f if line.strip()]
    if len(ids) != N_TRAIN_PAIRS or len(set(ids)) != N_TRAIN_PAIRS:
        return False
    return all((images / f"{i}.png").is_file() for i in ids)


def _download_from_hf(data_dir):
    from huggingface_hub import snapshot_download
    snapshot_download(
        HF_REPO,
        repo_type="dataset",
        allow_patterns=["images/*.png", "prompts.jsonl", "starterkit.zip"],
        local_dir=str(data_dir),
    )


def _extract_starterkit(data_dir, work_dir):
    kit_dir = work_dir / "starterkit"
    if (kit_dir / "sample_submission" / "model.py").is_file():
        return kit_dir
    archive = data_dir / "starterkit.zip"
    if not archive.is_file():
        from huggingface_hub import hf_hub_download
        hf_hub_download(HF_REPO, "starterkit.zip", repo_type="dataset", local_dir=str(data_dir))
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if not all(n.startswith("starterkit/") and ".." not in n for n in names):
            raise RuntimeError("starterkit.zip에 예상하지 못한 경로가 있습니다.")
        z.extractall(work_dir)
    return kit_dir


def prepare(use_drive=True, save_to_drive=True, work_dir=None):
    """데이터와 스타터 키트를 준비하고 주요 경로를 돌려준다.

    use_drive: Colab에서 Drive를 마운트하고, Drive에 저장된 데이터 묶음이 있으면 그것을 쓴다.
    save_to_drive: Drive에 데이터 묶음이 없으면 HF에서 받은 뒤 Drive에 저장한다(처음 한 번, 약 5GB).
    """
    env = detect_env()
    if work_dir is None:
        work_dir = {"colab": "/content/work", "kaggle": "/kaggle/working/work"}.get(env, "work")
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir = work_dir / "haikuthon-train"

    drive_on = env == "colab" and use_drive
    if drive_on:
        _mount_drive()
    drive_tar = DRIVE_DIR / DATA_TAR_NAME

    if data_ok(data_dir):
        source = "이미 준비됨"
    elif drive_on and drive_tar.is_file():
        print(f"Drive에서 데이터 묶음을 풉니다: {drive_tar}")
        with tarfile.open(drive_tar) as t:
            try:
                t.extractall(work_dir, filter="data")
            except TypeError:  # filter 인자가 없는 오래된 Python
                t.extractall(work_dir)
        source = "Drive"
    else:
        print(f"Hugging Face에서 받습니다: {HF_REPO}")
        _download_from_hf(data_dir)
        source = "Hugging Face"
        if drive_on and save_to_drive:
            print(f"다음 세션을 위해 Drive에 저장합니다: {drive_tar}")
            tmp = drive_tar.with_suffix(".tar.part")
            with tarfile.open(tmp, "w") as t:
                for name in ("prompts.jsonl", "images", "starterkit.zip"):
                    if (data_dir / name).exists():
                        t.add(data_dir / name, arcname=f"haikuthon-train/{name}")
            tmp.replace(drive_tar)

    if not data_ok(data_dir):
        raise RuntimeError(f"데이터 확인 실패: {data_dir}")
    kit_dir = _extract_starterkit(data_dir, work_dir)

    results_dir = (DRIVE_DIR / "results") if drive_on else (work_dir / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "env": env,
        "work_dir": work_dir,
        "data_dir": data_dir,
        "kit_dir": kit_dir,
        "results_dir": results_dir,
    }
    print(f"[{env}] 데이터 출처: {source}")
    for k, v in paths.items():
        if k != "env":
            print(f"  {k:12s} {v}")
    return paths


def copy_starterkit(kit_dir, dest):
    """train.py는 자기 폴더의 sample_submission/models를 덮어쓴다. 원본을 지키려고 복사본에서 실행한다."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(kit_dir, dest, ignore=shutil.ignore_patterns("sample_submission.zip", "__pycache__"))
    return dest


def env_info():
    import platform
    info = {"env": detect_env(), "python": platform.python_version(), "cpu_count": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_mem_gb"] = round(p.total_memory / 1024**3, 1)
            info["visible_gpus"] = torch.cuda.device_count()
    except ImportError:
        info["torch"] = None
    for pkg in ("transformers", "safetensors", "sentencepiece", "PIL", "huggingface_hub"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = None
    return info

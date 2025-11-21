#!/usr/bin/env python3
import os
import sys
import urllib.request
import time
import shutil

# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

BASE_URL = "https://files.ipd.uw.edu/pub/rfdiffusion2/"
BASE_FS = "/net/lab/pub/rfdiffusion2/"

# Explicit files whose dest name/path differs from URL or that aren't in the weights pattern
FILES = [
    {
        "url": BASE_URL + "sifs/rfdiffusion.sif",
        "dest": os.path.join("rf_diffusion", "exec", "bakerlab_rf_diffusion_aa.sif"),
    },
    {
        "url": BASE_URL + "sifs/mlfold.sif",
        "dest": os.path.join("rf_diffusion", "exec", "mlfold.sif"),
    },
    {
        "url": BASE_URL + "sifs/chai.sif",
        "dest": os.path.join("rf_diffusion", "exec", "chai.sif"),
    },
]

# Minimal specification for weights/third-party weights:
# Just list their RELATIVE paths under the base (everything after .../pub/rfdiffusion2/)
WEIGHTS = [
    "model_weights/RFD_173.pt",
    "model_weights/RFD_140.pt",
    "third_party_model_weights/ligand_mpnn/s25_r010_t300_p.pt",
    "third_party_model_weights/ligand_mpnn/s_300756.pt",
]

# Expand WEIGHTS into full FILES entries (same format as your current FILES)
for rel in WEIGHTS:
    FILES.append({
        "url": BASE_URL + rel,
        "dest": os.path.join("rf_diffusion", *rel.split("/")),
    })

ARGS = set(sys.argv[1:])
OVERWRITE = "overwrite" in ARGS
COPY_WITHIN_DIGS = "copy_within_digs" in ARGS

def ensure_dest_ready(dest):
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.islink(dest):
        print(f"{Colors.YELLOW}⚠ Removing existing symlink:{Colors.RESET} {dest}")
        os.remove(dest)
    elif os.path.isfile(dest) and not OVERWRITE:
        print(f"{Colors.CYAN}✔ File exists, skipping:{Colors.RESET} {dest} "
              f"(use '{Colors.BOLD}overwrite{Colors.RESET}' to replace)")
        return False
    return True

def rel_from_url(url: str) -> str:
    """Return path relative to BASE_URL given a full URL."""
    if not url.startswith(BASE_URL):
        # Fallback: try to treat it as already-relative (shouldn't happen here)
        return url
    return url[len(BASE_URL):]

def copy_file_within_digs(url, dest):
    rel = rel_from_url(url)
    src = os.path.join(BASE_FS, *rel.split("/"))
    if not os.path.isfile(src):
        print(f"{Colors.RED}✖ Source not found:{Colors.RESET} {src}")
        return
    print(f"{Colors.BLUE}⇢ Copying (within digs):{Colors.RESET} {src}")
    print(f"{Colors.BLUE}→ To:{Colors.RESET} {dest}")
    total = os.path.getsize(src)
    copied = 0
    chunk = 1024 * 1024
    start = time.time()
    def _eta_str(elapsed, done, total_bytes):
        remaining = max(0, total_bytes - done)
        speed = done / elapsed if elapsed > 0 else 0
        if speed <= 0:
            return "?"
        eta = int(remaining / speed)
        m, s = divmod(eta, 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}"
    def _human(n: float) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024.0:
                return f"{n:3.1f}{unit}"
            n /= 1024.0
        return f"{n:.1f}PB"
    try:
        with open(src, "rb") as fsrc, open(dest, "wb") as fdst:
            while True:
                buf = fsrc.read(chunk)
                if not buf:
                    break
                fdst.write(buf)
                copied += len(buf)
                pct = (copied / total) * 100 if total > 0 else 0
                elapsed = time.time() - start
                speed = copied / elapsed if elapsed > 0 else 0
                eta = _eta_str(elapsed, copied, total)
                sys.stdout.write(
                    f"\r{Colors.CYAN}{pct:5.1f}%{Colors.RESET} "
                    f"{_human(copied)} / {_human(total)} "
                    f"({_human(speed)}/s) ETA {eta}"
                )
                sys.stdout.flush()
        shutil.copystat(src, dest)
        sys.stdout.write("\n")
        print(f"{Colors.GREEN}✓ Copy complete:{Colors.RESET} {dest}\n")
    except Exception as e:
        sys.stdout.write("\n")
        print(f"{Colors.RED}✖ Copy error:{Colors.RESET} {e}")

def download_file(url, dest):
    def _human(n: float) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024.0:
                return f"{n:3.1f}{unit}"
            n /= 1024.0
        return f"{n:.1f}PB"

    def _eta_str(elapsed, done, total_bytes):
        remaining = max(0, total_bytes - done)
        speed = done / elapsed if elapsed > 0 else 0
        if speed <= 0:
            return "?"
        eta = int(remaining / speed)
        m, s = divmod(eta, 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}"

    print(f"{Colors.BLUE}⇣ Downloading:{Colors.RESET} {url}")
    print(f"{Colors.BLUE}→ To:{Colors.RESET} {dest}")
    start = time.time()

    def _reporthook(blocknum, blocksize, totalsize):
        downloaded = blocknum * blocksize
        if totalsize > 0:
            if downloaded > totalsize:
                downloaded = totalsize
            pct = downloaded / totalsize * 100
        else:
            pct = 0.0

        elapsed = time.time() - start
        speed = downloaded / elapsed if elapsed > 0 else 0
        eta = _eta_str(elapsed, downloaded, totalsize) if totalsize > 0 else "?"

        if totalsize > 0:
            sys.stdout.write(
                f"\r{Colors.CYAN}{pct:5.1f}%{Colors.RESET} { _human(downloaded) } / { _human(totalsize) } "
                f"({ _human(speed) }/s) ETA {eta}"
            )
        else:
            sys.stdout.write(
                f"\r{Colors.CYAN}{_human(downloaded)}{Colors.RESET} ({ _human(speed) }/s)"
            )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, _reporthook)
        sys.stdout.write("\n")
        print(f"{Colors.GREEN}✓ Download complete:{Colors.RESET} {dest}\n")
    except Exception as e:
        sys.stdout.write("\n")
        print(f"{Colors.RED}✖ Error during download:{Colors.RESET} {e}")

def transfer_file(url, dest):
    if not ensure_dest_ready(dest):
        return
    try:
        if COPY_WITHIN_DIGS:
            copy_file_within_digs(url, dest)
        else:
            download_file(url, dest)
    except Exception as e:
        print(f"{Colors.RED}✖ Error:{Colors.RESET} {e}")

def main():
    mode = "COPY (within digs)" if COPY_WITHIN_DIGS else "DOWNLOAD"
    print(f"{Colors.HEADER}{Colors.BOLD}=== Setting up RFDiffusion environment ==={Colors.RESET}")
    print(f"{Colors.BOLD}Mode:{Colors.RESET} {mode} {'(overwrite enabled)' if OVERWRITE else ''}\n")
    for f in FILES:
        transfer_file(f["url"], f["dest"])
    print(f"{Colors.GREEN}{Colors.BOLD}All files ready! 🎉{Colors.RESET}")

if __name__ == "__main__":
    main()

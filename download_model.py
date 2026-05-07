import sys

def download_silero_vad(retries: int = 3) -> None:
    import torch
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading Silero VAD model (attempt {attempt}/{retries})...")
            torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                source="github",
                force_reload=False,
                trust_repo=True,
            )
            print("Silero VAD model cached successfully.")
            return
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}", file=sys.stderr)
            if attempt == retries:
                print(
                    "WARNING: Could not pre-cache Silero VAD model. "
                    "It will be downloaded at runtime.",
                    file=sys.stderr,
                )

if __name__ == "__main__":
    download_silero_vad()

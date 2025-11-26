import subprocess
import sys


def install_requirements():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "openai>=1.0.0", "requests>=2.28.0"]
    )


if __name__ == "__main__":
    install_requirements()

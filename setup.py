"""Setup script для установки пакета transcribator."""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README для длинного описания
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="transcribator",
    version="0.1.0",
    description="Инструмент для транскрибации видео с использованием OpenAI Whisper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    packages=find_packages(),
    install_requires=[
        "openai-whisper>=20231117",
        "click>=8.0.0",
        "tqdm>=4.66.0",
        "librosa>=0.10.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "transcribator=transcribator.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

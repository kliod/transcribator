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
    include_package_data=True,
    package_data={
        "transcribator": ["templates/*.html"],
    },
    install_requires=[
        "openai-whisper>=20240930",
        "faster-whisper>=1.2.1",
        "ctranslate2>=4.7.0",
        "click>=8.0.0",
        "tqdm>=4.66.0",
        "librosa>=0.10.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "fastapi>=0.128.0",
        "uvicorn>=0.40.0",
        "jinja2>=3.1.0",
        "python-multipart>=0.0.20",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "transcribator=transcribator.cli:main",
            "transcribator-web=transcribator.webapp:main",
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

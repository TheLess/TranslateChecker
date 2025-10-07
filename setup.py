"""
Translation Checker 安装脚本
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 读取requirements文件
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="translation-checker",
    version="1.0.0",
    author="Translation Checker Team",
    author_email="support@translationchecker.com",
    description="一个基于机器学习的翻译质量检查工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/translation-checker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Quality Assurance",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "local-llm": [
            "llama-cpp-python>=0.2.0",
            "ctransformers>=0.2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "translation-checker=main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "examples/*.xlsx"],
    },
)

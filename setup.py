from setuptools import setup, find_packages

setup(
    name="phishing_URL_detection",
    version="0.1.0",
    description="A data science project for detecting phishing URLs",
    author="Polina Polskaia",
    author_email="polskaia@bu.edu",
    url="https://github.com/polinacsv/phishing_URL_detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        # empty for now, dependencies are tracked in requirements.txt
    ],
)
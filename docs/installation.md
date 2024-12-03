# Installation Guide

Follow the steps below to set up and install the `japl` project from the repository.
>for Windows users make sure to use either Powershell terminal or GitBash

## Prerequisites

1. **Python Environment**:  
   It is recommended to use a **Conda** environment for managing dependencies and isolating your Python environment. You can download and install Conda from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Git**:  
   Ensure Git is installed to clone the project repository. You can download it from [git-scm.com](https://git-scm.com/).

---

## Step 1: Clone the Repository

First, clone the project repository from GitHub:

```bash
git clone https://github.com/nobono/japl.git
cd japl
```

---

## Step 2: Set Up a Conda Environment
Create and activate a new Conda environment:

```bash
conda create -n [ENV_NAME] python=3.11 -y
conda activate [ENV_NAME]
```
>where `ENV_NAME` is whatever conda environment name you choose

---

## Step 3: Install Dependencies
Install the required Python packages listed in `required.txt`:

```bash
pip install -r requirements.txt
```

---

## Step 4: Install the Package
Install the project using `setup.py`:

```bash
python setup.py install
```

---

## Step 5: Verify Installation
To confirm the installation was successful, run:

```bash
japl
```

where you should see the output:
```bash
=============================================
                    JAPL
        Just Another Prototyping Layer
=============================================
```

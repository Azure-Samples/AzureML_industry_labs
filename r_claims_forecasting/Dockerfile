# ── R Claims Severity Forecasting: Custom Azure ML Environment ───
#
# Builds a Docker image with R 4.x and Python 3.10 (conda) for running
# R Gamma GLM inference via the Azure ML batch endpoint driver.
#
# Base image: Azure ML OpenMPI Ubuntu 22.04 (includes conda Python).
# R is installed from CRAN without software-properties-common to avoid
# a system Python that would shadow the conda interpreter.

FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

# ── Install R (without system Python) ────────────────────────────
# IMPORTANT: We avoid installing software-properties-common because it
# pulls in system Python 3.10 (/usr/bin/python3) which shadows the conda
# Python used by the Azure ML batch inference driver.  Instead we add
# the CRAN repo manually via apt-key (the deprecation warning is benign).
RUN apt-get update && \
    apt-get install -y --no-install-recommends dirmngr gnupg2 && \
    # apt-key is deprecated but still functional; the warning is expected
    apt-key adv --keyserver keyserver.ubuntu.com \
        --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    echo "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
        > /etc/apt/sources.list.d/cran.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        r-base-core \
        r-base-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev && \
    rm -rf /var/lib/apt/lists/*

# ── Install R packages ───────────────────────────────────────────
# argparse: CLI argument parsing for pipeline step scripts
# jsonlite: JSON I/O for metrics and normalisation stats
RUN Rscript -e "install.packages(c('argparse', 'jsonlite'), repos='https://cloud.r-project.org/')"

# ── Install Python dependencies ──────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Verify pkg_resources is importable (required by Azure ML batch driver) ──
# The base image ships setuptools>=82 which removed pkg_resources.
# requirements.txt pins setuptools<82 to restore it.
RUN python -c "import pkg_resources; print('pkg_resources OK')"

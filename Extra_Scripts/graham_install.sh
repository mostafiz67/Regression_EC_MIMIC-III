#!/bin/bash
if [[ -z "${CC_CLUSTER}" ]];
then
    version=$(python --version | cut -d " " -f 2)
    major=$(echo "$version" | cut  -d '.' -f 1)
    minor=$(echo "$version" | cut  -d '.' -f 2)
    if test "$major" -lt 3
    then
        echo "Python 3.8 required"
        exit 1
    fi
    if test "$minor" -lt 8
    then
        echo "Python 3.8 required"
        exit 1
    fi
    echo "Not on Compute Canada. Using local python: $(python --version)"
    python -m venv .venv
else
    echo "On Compute Canada Graham. Working around python/3.8 issues..."
    module load gcc/9.3.0 arrow python/3.8 scipy-stack
    echo "Using $(python --version)"
    echo -n "Importing required modules... "
    echo "done."
    virtualenv --no-download .venv
fi

source .venv/bin/activate
pip install setuptools==59.5.0 && \
pip install \
    joblib \
    matplotlib \
    numba \
    numpy \
    pandas \
    pyarrow \
    pytest \
    pytorch_lightning==1.5.1 \
    scipy \
    scipy \
    seaborn \
    sklearn \
    tabulate \
    torch \
    torchvision \
    torchaudio \
    torchmetrics \
    tqdm \
    typing_extensions \
    wfdb \
    scikit-image

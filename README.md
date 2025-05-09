- [Project Tasks](#project-tasks)
  - [Challenges](#challenges)
- [Data Flow, Pre-Processing, and Decimation](#data-flow-pre-processing-and-decimation)
  - [Data Flow](#data-flow)
  - [Decimation](#decimation)
- [Useful References](#useful-references)

# Project Tasks

Broadly, the goal here is to assess the viability of ML and Deep Learning (DL) tools for predicting
irregular / low-density / rare and serious clinical events (including costly lab measurements, need
for procedures, fatalities, and others) from continuously-available / high-density / regular
monitoring data (e.g. EKG, ABP, PPG, 02, etc).

We hope to do this using the rich data available from Physionet, in particular the [MIMIC-IV
Clinical Database](https://physionet.org/content/mimiciv/1.0/) for clinical events, and the
[MIMIC-III Waveform Database](https://physionet.org/content/mimic3wdb/), and [MIMIC-III Waveform
Database Matched Subset](https://physionet.org/content/mimic3wdb-matched/) for continuous monitoring
data.


## Challenges

The challenges of this project are engineering / technical than modeling / theoretical.

The most successful cases of deep learning (image classification/segmentation, NLP),
involve extremely clear, noise-free relationships between inputs and outputs. An image with
a dog in it (even if distorted and including other animals) is such with near 100% certainty: if
that image is shown to a group of a thousand human raters *that have seen a dog before and are
answering honestly*, then we expect basically 0 raters to say "there is no dog here" (and in this
case we suspect / infer that the problem is rather the rater than the input).

Or, put in causal terms, the causal relationship between the input (image with a dog) and output
(either dog classification probability, or segmentation map of pixels that include dog, or NLP
translation) is near 100% certain (there will be very little disagreement between a group of human
beings that classify the picture / pixels as "dog" or "not dog"). Or again put another way, the
image really does have all the information needed to create an accurate and meaningful output, and
we all mostly recognize that performance failures of these models is ***due to model and training
limitations and not fundamental causal uncertainties or noise issues in the data itself***. That is,
the problem is not the data, but that the models are too simple, or that a complex-enough model
can't currently be trained effectively enough.

However, for our data, the problem is somewhat reversed. In most cases, we expect that the causal
relationship between predictor and target is weak, or, equivalently, that the predictor always
includes substantial noise (subject-level factors, predictors are epiphenomena rather than
biologically-predictive primitives, targets are also epiphenomena and multiply-caused). Thus, we do
not expect to need particularly flexible models. I.e. while in general we should not expect the
basic relationships between predictor(s) and target to be fundamentally linear, it should be hard to
beat linear models due to noise / confounding / causal impurity.

Instead, the irregularity and sizes of the data limit its utility in both classical and DL models.
I.e. the data is large enough that only models that can be fit sequentially (e.g. via batches as in
stochastic gradient descent) are computationally viable, and the data is complex enough
(subject-level, irregularly-sampled, variable input and output sizes) that it is not even clear how
to even evaluate classical models on the data.




the basic deep learning architectures available today


# Data Flow, Pre-Processing, and Decimation

## Data Flow

Our data goes from raw to "usable for deep-learning" (DL-usable, i.e. windowizable and index-mapped
for `__getitem__` calls) in many steps. There are thus a number of states and transitions
(processing steps) for our data to pass through before states of the data:

| Step | State            | Location    | Disk Memory    | RAM    | Filetypes                    | Code                                |
|------|------------------|-------------|----------------|--------|------------------------------|-------------------------------------|
| 0    | RAW ONLINE       | cloud       | 2TB            | >2TB   | .hea (metadata), .dat (wave) | `scripts/shell`                     |
| 1    | RAW DOWNLOADED   | Niagara     | 1-2TB          | >2TB   | .hea (metadata), .dat (wave) | `scripts/shell`                     |
| 2    | META-DATA PARSED | Niagara     | few GBs        | <1GB   | .hea (metadata) only         | `src/acquisition`                   |
| 3    | CONDENSED        | any cluster | 100GB to 2TBs* | >100GB | .parquet () only             | `src/acquisition`, `src/preprocess` |
| 4    | EDGE-NAN CLEANED | any cluster | 100GB to 2TBs* | >100GB | .parquet () only             | `src/acquisition`, `src/preprocess` |
| 5    | FULLY CLEANED    | any cluster | ? GB to 2TBs*  | ? GB   | .parquet () only             | **TODO**                            |
| 6    | SMOOTH DECIMATED | any cluster | 20GB to 2TBs** | ? GB   | .parquet () only             | `src/preprocess`                    |
| 7    | INDEX MAPPED     | in RAM      | -              | <180GB | -                            | `deepsubject.py`, `dataloader.py`   |

\* Size depends on number of waveform modalities extracted

\*\* size depends on decimation choice

Each step / state except step 0 has an associated processing cost:

| Step | State            | Processing Step | Processing Time   | Needs Parallel  | Needs GPU |
|------|------------------|-----------------|-------------------|-----------------|-----------|
| 0    | RAW ONLINE       | -               | -                 | -               |           |
| 1    | RAW DOWNLOADED   | Downloading     | ~24 hours         | NO              | NO        |
| 2    | META-DATA PARSED | parsing         | ~30 min (Niagara) | YES             | NO        |
| 3    | CONDENSED        |                 | ~1hr              | YES (~20 cores) | NO        |
| 4    | EDGE-NAN CLEANED |                 | ~0                | NO              | NO        |
| 5    | FULLY CLEANED    |                 | ???               | YES             | NO        |
| 6.1  | SMOOTH DECIMATED | Smoothing       | hours             | YES             | NO        |
| 6.2  | SMOOTH DECIMATED | Decimation      | ~0                | NO              | NO        |
| 7    | INDEX MAPPED     | Window counting | <1hr              | NO              | YES       |

Steps 5 through 7 have hyperparameters we may want to tune, and so code can't make any assumptions
about the values of those hyperparameters. E.g. all code will always need to read from a config including
various full-cleaning, smoothing, decimation, and windowing options.

## Decimation

Our predictor data is generally sampled at a high rate (125 Hz or more for ABP, EKG, etc data, lower
for numerics). Our targets (chart or lab events) are generally sampled irregularly and highly
infrequently (a few times per day or hour). Thus we will want to smooth and decimate for a number of reasons:

- *a priori* it is highly unlikely the full sampling rate is useful for predicting much more rarely
  sampled targets
- the full data without decimation is likely too computationally and memory intensive to use
  practically
- even if neither of the above is true, there are likely tradeoffs between efficiency and performance, and
  we want to be able to optimize decimation and smoothing as hyperparameters anyway
- it is possible we may also want to design different models for different pre-processed variants of
  the data (e.g. architecture kernel sizes)

While we can certainly *start* by setting up code to use a small handful of pre-decimated and
pre-smoothed copies of the data (e.g say decimations of 1, 5, 25, 125, 250, 500, plus some smoothing
methods), code still needs to be able to operate in general for arbitrary decimation choice.


We have to work with the raw (un-decimated) data

# Useful References

- other MIMIC III prediction benchmarks with a Transformer architecture: https://arxiv.org/abs/1711.03905

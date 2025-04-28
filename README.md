# CS526 Final Project: Highway Contract Bid Price Prediction

## Overview

### Data

- [Dataverse Link](https://doi.org/10.7910/DVN/DNIWOA)

### Video

- **[Video Walkthrough of Project/Repo](https://youtu.be/Qmv7ebjDKWM)**

### Checklist

- Final Project Grading Rubric
  - Problem Definition
    - In Jupyter Notebook _BLUF_: Highway Contractors bid on state DoT released contracts. Each item of the contract must be bid on. Lowest total bid wins. Estimators working for contracting companies want to be the lowester bidder but not so low that they leave significant money on the table. Additionally, they want to streamline the bidding process (assembly of this data typically takes weeks).
  - Data Choice & Fit
    - Directly from North Caroline Department of Transporation Website. They release all data since these are goverment contracts. Blessing and a curse because there is a lot of data but not all of it is necessary == lots of parsing.
  - Scope Alignment
    - Project parses historical data and uses KNN Regression to predict what a contractor will bid for each payitem on a new contract (extremely applicable!).
  - Exploratory Data Analysis
    - I do a cluster analysis based on the embedding of the textual information in the contracts. Since this was inspired by my internship this past summer, I used the information I learned during that time to narrow the scope to the most influential variables (based on what estimators mainly connsider during the manual bidding process.)
  - Hypothesis & Logic
    - Discussed throughout the notebook.
  - Statistical Rigor
    - Discussed throughout the notebook.
  - Insight & Interpretation
    - I discuss the methods and used and why. This is where I relied on the knowledge I gained through discussions with a former estimator on the bidding process.
  - Failure Analysis
    - Touched on in the conclusion of the notebook. Came up short on time given the necessary data collection and preprocessing. I discuss the methods I would have explored if I had more time.
  - Tool Selection
    - I relied on python because of its extensive libraries, preference in the ML community, and quick proof of concept development cycle. I also used [`uv`](https://docs.astral.sh/uv/) as my package manager, which is a modern package manager written in Rust and gaining traction in the python community.
  - Distributed/HPC Resources
    - No heavy computations so I did not use any distributed/HPC resources.
  - Version Control
    - Used git and github for version control. Should've commited more frequently at the beggining of the project.
  - Environment Management
    - Used `uv` to manage the project. My approach was to create my modules [located in `src/`] in pure python and then have a main jupyter notebook to piece the modules together and test my code.
  - Code Quality
    - Place code in module files and imported them into the jupyter notebook. A little lengthy with some function; largely attributed to timeline.
  - Functionality
    - The code works if you have API keys for [Google Maps](https://developers.google.com/maps/documentation/javascript/get-api-key) and [JINA](https://jina.ai/). I removed those since this is a public repo and I do not want to expose my API keys.
  - Data Handling
    - I used `selenium` to scrape my data from the NCDOT website and used `pandas` and `csv` for most of the preprocessing.

## Usage

```bash
# Clone the repository to local directory
git clone
cd CS526Final
# Set up project environment
uv sync
# Open Jupyter Notebook and Run Cells
jupyter notebook Wetzel_CS526Final.ipynb
```

## Contact

- Calvin Wetzel
- jwetzel2@vols.utk.edu

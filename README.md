# ML Retraining Scheduling Research

**When Is Reinforcement Learning Worthwhile for Model Retraining? Evidence from Synthetic Drift, Cloud Prices, and Electricity Data**

This repository contains the Python implementations, datasets, and experimental results for an undergraduate research study comparing different machine learning (ML) model retraining strategies.

The main objective is to investigate whether reinforcement learning can make better retraining decisions than traditional scheduling methods when model accuracy decreases and cloud computing prices change over time.

## 1. Research Objectives

This research aims to:

- Compare traditional and reinforcement learning-based retraining strategies.
- Investigate how concept drift affects deployed ML models.
- Evaluate the impact of changing cloud computing prices on retraining decisions.
- Compare retraining costs, prediction accuracy, and overall net utility.
- Evaluate scheduling methods using synthetic data and a real-world dataset.

## 2. Retraining Strategies

The following strategies are evaluated:

| Strategy | Description |
|---|---|
| Never Retrain | Keeps the original model without retraining |
| Always Retrain | Retrains at every simulation step |
| Fixed Interval | Retrains after a predefined number of steps |
| Accuracy Threshold | Retrains when model accuracy decreases beyond a threshold |
| Price-Aware Threshold | Adjusts the accuracy threshold based on current computing prices |
| Q-learning | Uses tabular reinforcement learning to select retraining actions |
| Deep Q-Network (DQN) | Uses a neural network to learn retraining decisions |
| Sampled CARA-T | An approximate implementation of a cost-aware retraining baseline |

The sampled CARA-T implementation is an adaptation and should not be considered an exact reproduction of the original CARA algorithm.

## 3. Datasets

### Synthetic Concept Drift

A synthetic classification dataset is generated to simulate three types of concept drift:

- **Sudden drift:** The data distribution changes unexpectedly.
- **Gradual drift:** The data distribution changes slowly over time.
- **Recurring drift:** Previously observed data patterns return.

Each simulation episode contains 200 time steps, with 500 observations per step.

### AWS EC2 Spot Price History

Historical AWS EC2 Spot prices are used to represent changing cloud computing costs.

- Instance type: `g4dn.xlarge`
- Region: `us-east-1`
- Availability zone: `us-east-1a`
- Period: September 16 – October 16, 2026
- Hourly price observations: 721

The collected prices are replayed during experiments. Actual GPU retraining jobs were not executed on AWS.

### Electricity (ELEC2) Dataset

The Electricity dataset is a real-world temporal classification dataset commonly used in concept drift research.

- Total records: 45,312
- Initial training records: 27,187
- Validation records: 9,062
- Exploratory test records: 9,063

The dataset is processed chronologically to preserve the order of observations.

## 4. Project Structure

```text
ml-retraining-scheduling-research/
├── datasets/
│   ├── electricity.csv
│   └── hourly_prices.csv
│
├── experiments/
│   ├── retrain_sim.py
│   ├── aws_prices.py
│   ├── prepare_prices.py
│   ├── electricity_retraining.py
│   ├── electricity_rl_env.py
│   ├── electricity_walkforward.py
│   ├── electricity_walkforward_rl.py
│   └── ...
│
├── results/
│   ├── electricity_walkforward_folds.csv
│   ├── electricity_walkforward_choices.csv
│   ├── electricity_walkforward_rl_folds.csv
│   └── ...
│
├── paper/
│   └── research_paper.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

The structure above is the intended repository layout. Additional experiment scripts, model checkpoints, and result files should be included where available.

## 5. Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ml-retraining-scheduling-research.git
cd ml-retraining-scheduling-research
```

Replace `YOUR_USERNAME` with your GitHub username.

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include commonly used packages such as NumPy, Pandas, Scikit-learn, Gymnasium, Stable-Baselines3, and Boto3.

The exact package versions should be recorded in `requirements.txt`.

## 6. Running the Experiments

### Synthetic Concept Drift Experiments

The synthetic experiments evaluate different scheduling strategies under changing data distributions and simulated computing prices.

The simulation implementation is located in:

```text
experiments/retrain_sim.py
```

Use the experiment runner scripts included in the repository to generate the synthetic evaluation results.

### Historical AWS Spot Price Experiments

To retrieve historical pricing data:

```bash
python experiments/aws_prices.py
```

AWS credentials and relevant EC2 permissions are required for this data-collection step.

To prepare the hourly price dataset:

```bash
python experiments/prepare_prices.py
```

For reproducibility, use the saved `hourly_prices.csv` file when repeating the original experiments. A fresh AWS query may return different historical data.

### Electricity Experiments

Run the Electricity retraining experiment:

```bash
python experiments/electricity_retraining.py
```

Run the chronological walk-forward evaluation:

```bash
python experiments/electricity_walkforward.py
```

Run the walk-forward Q-learning and DQN evaluation:

```bash
python experiments/electricity_walkforward_rl.py
```

The final RL experiment should use the training settings recorded in the experiment script. Short smoke-test configurations are not equivalent to the reported research results.

## 7. Evaluation Metrics

Scheduling strategies are evaluated using:

- **Prediction accuracy:** How accurately the deployed model predicts incoming data.
- **Retraining cost:** The total estimated computing expense of retraining.
- **Number of retrainings:** How frequently the scheduling policy updates the model.
- **Net utility:** The value of prediction accuracy minus retraining cost.
- **Accuracy improvement per dollar (APD):** The improvement in model accuracy relative to a never-retrain baseline, compared with retraining expenditure.

Net utility is the primary evaluation metric.

## 8. Main Experimental Results

### Synthetic Experiments

The tuned price-aware threshold achieved higher mean net utility than tabular Q-learning at all three tested retraining cost levels.

### Historical AWS Price Replay

| Strategy | Mean Net Utility |
|---|---:|
| Accuracy Threshold | 3699.852 |
| Price-Aware Threshold | 3699.852 |
| DQN | 3699.708 |
| Q-learning | 3698.656 |
| Sampled CARA-T | 3676.683 |

Utilities are reported per 200-step synthetic episode under the recorded-price replay.

### Electricity Walk-Forward Evaluation

| Strategy | Net Utility per Batch | Accuracy |
|---|---:|---:|
| Fixed Interval | **12.3150** | **64.51%** |
| Price-Aware Threshold | 12.0975 | 62.15% |
| Accuracy Threshold | 12.0975 | 62.15% |
| DQN | 11.3503 | 56.79% |
| Q-learning | 11.3385 | 56.71% |
| Never Retrain | 11.3385 | 56.71% |

The results show that the best-performing strategy depends on the dataset, retraining cost, and evaluation conditions.

These findings are exploratory and do not establish that one strategy is statistically superior across all environments.

## 9. Reproducibility

The experiments use fixed training configurations and random seeds where specified in the scripts.

To reproduce the results:

1. Install the recorded Python dependencies.
2. Use the original saved datasets and AWS price history.
3. Run the experiments using the documented training settings.
4. Save individual training-seed and fold-level results.
5. Compare the outputs against the results reported in the paper.

Some experiments share the same historical AWS pricing sequence, and the Electricity dataset was examined during development. These limitations should be considered when interpreting the results.

## 10. Research Limitations

The research has several limitations:

- GPU retraining costs are estimated using assumed job durations.
- Historical AWS pricing covers a limited period and availability zone.
- Reinforcement learning experiments use a limited number of independent training seeds.
- The original synthetic simulation makes simplifying assumptions about data availability and retraining time.
- The Electricity evaluation is exploratory.
- The sampled CARA-T baseline is not an exact implementation of the published algorithm.

Future research could evaluate additional datasets, longer cloud pricing periods, measured retraining costs, and more reinforcement learning training runs.

## 11. Research Paper

The IEEE-format research paper is available in the `paper/` directory.

**Title:** *When Is Reinforcement Learning Worthwhile for Model Retraining? Evidence from Synthetic Drift, Cloud Prices, and Electricity Data*

## 12. Author

**Dilum Piumantha**

Department of Computer Engineering  
General Sir John Kotelawala Defence University (KDU)  
Sri Lanka

## 13. Acknowledgment

This repository was developed as part of an undergraduate research study on cost-aware machine learning model retraining.

The research uses the Electricity dataset, historical AWS EC2 Spot prices, and established machine learning and reinforcement learning methods.

## 14. Disclaimer

The reported monetary utility values are calculated using an experimental cost model. They should not be interpreted as measured business revenue or actual cloud deployment savings.

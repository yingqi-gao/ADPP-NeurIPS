# Market Design of Machine Learning Data Marketplace

This repository is the official implementation of [Market Design of Machine Learning Data Marketplace](https://openreview.net/forum?id=TA4uUUuRlq). 

![ADPP flowchart](./assets/ADPP.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the RSRDE(s) in the paper, run this command:

```train
python train.py <distribution_type>
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.


## Test
To generate the test data and the ideal revenue benchmark in the paper, run this command:

```test
python test.py <distribution_type>
```

## Evaluation

To calculate the average regrets involved in the following 4 mechanisms - DOP, RSOP, RSKDE, and RSRDE, run:

```eval
python eval.py <distribution_type> <mechanism> <#training_bids> <#training_rounds>
```

To plot the regrets and compare visually, run:

```eval
python plot.py <distribution_type>
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Results

Our model achieves the following performance on :
Uniform                    |Normal
:-------------------------:|:-------------------------:
![Boxplots of average regrets for uniform distribution family](./assets/uniform.png) | ![Boxplots of average regrets for normal distribution family](./assets/normal.png)

Exponential                |Real
:-------------------------:|:-------------------------:
![Boxplots of average regrets for exponential distribution family](./assets/exponential.png) | ![Boxplots of average regrets for real FCC AWS-3 data](./assets/real.png)

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 



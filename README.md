# EEG_Recording_Patterns

### Authors: Joshua Thompson and Matthew Carroll

### Goal
#### Primary Goal
Find any frequency at any channel that predicts theta (evoked or induce) power at FCz or are predicted by theta at FCz:
    1. Directly using power data
    2. Not separating evoked/induced power which is averaged over trials, but using total power in trial by trial sense. 
Note: need to distinguish between conditions, especially between reward and no reward.

#### Secondary Goal
Try to find the relationship between EEG data recorded after feedback, and if there is anyway we can use it to predict behavior (choice and/or reaction time). Theta is involved somewhere, but perhaps we can find another frequency at another electrode that can predict whether the subject will make a left choice vs a right choice on the next trial, and whether these oscillations can predict whether the subject will win-stay (following positive feedback) or lose-shift (following negative feedback).

### Data
Private dataset collected via a 36-electrode EEG dataset from 135 college-age students. Each subject was sampled 200 times at a 250 Hz sample rate. 

Initial data shaped as (135, 120, 1250) matrix.
    * 135 subjects
    * 120 frequencies
    * 1250 time samples

1250 time samples:
    1. Button Press (1.5 seconds) - 371-372 time step
    2. Feedback Onset (~2.5 seconds, 1 second after button press) - starts at 626 time step
    3. Next trial (~3.5 seconds, 1 second after Feedback Onset) - starts at 888 time step
Therefore, in order to use the feedback onset we use time steps 626-888.

### Preprocessing
Because the data was raw, a substantial amount of [preprocessing][preprocess.py] needed to be done.
    1. Make the time data stationary
    2. Apply a Box Cox transformation
    3. Calculate difference to ensure stationality

Once this was done, the correlations between frequencies was compared to the theta frequency and the highest correlation was chosen. This was saved as testing/training datasets.

### Models
Various types of models were attempted. In addition to the currently working [models][train_ml.py] a transformer and LSTM-FCN were under development at the time of this writing. 
    1. RNN
    2. GRU
    3. LSTM

### How to Run
    1. Clone repo - required libraries are stored in requirements.txt
    2. Uncomment the model you wish to use in [train_ml][train_ml.py]. Make sure to uncomment the corresponding path as well.
    3. Run the command `python train_ml.py`
    4. Edit [eval_ml][eval_ml.py] to look for the correct model path
    5. Run the command `python eval_ml.py`

### Results
Overall, the baselineGRU model performed the best in predicting targeted values where RNN performed the worst. The graphs are shown below:

![RNN Test Predictions versus Targets](/graphs/RNN_test_preds_vs_targets.png)
The RNN Testing Dataset Predicition Versus Targets

![RNN Train Predictions versus Targets](/graphs/RNN_train_preds_vs_targets.png)
The RNN Training Dataset Prediction Versus Targets

![GRU Test Predictions versus Targets](/graphs/GRU_test_preds_vs_targets.png)
The GRU Testing Dataset Predicition Versus Targets

![GRU Train Predictions versus Targets](/graphs/GRU_train_preds_vs_targets.png)
The GRU Training Dataset Prediction Versus Targets

![LSTM Test Predictions versus Targets](/graphs/LSTM_test_preds_vs_targets.png)
The LSTM Testing Dataset Predicition Versus Targets

![LSTM Train Predictions versus Targets](/graphs/LSTM_train_preds_vs_targets.png)
The LSTM Training Dataset Prediction Versus Targets

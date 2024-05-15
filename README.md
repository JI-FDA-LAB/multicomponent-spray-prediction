# Deep learning framework for multicomponent-spray-prediction
This repo presents a deep learning framework, the Encoder-LSTM-Decoder structure, for spray cross-pattern prediction from the work *"Prediction of Spray Collapse Patterns on Different Fuel Mixture Compositions Based on Deep Learning Framework"*. It is realized in Python. This work indicated that the proposed framework could efficiently capture the underlying correlations between varying fuel mixture compositions and their resultant spray patterns given certain environmental parameters. This predictive capability facilitates the identification of optimal mixture states for achieving superior fuel atomization quality.

The source code for our model training, predicting, and testing is published in this GitHub repo. The sample dataset is uploaded in this repo as well. If you are interested in further expanding our idea to other pattern datasets, please cite our coming paper.

## Engine Parameters

## Computational Resources

## Repo Guidelines
The sample dataset, result, and the well-trained models are published but could be available on request.
### Sample Dataset
For both extrapolated and interpolated prediction, we use the **octane-fixed interpolation/extrapolation model** to demonstrate our work in this repo.
All the images in the dataset are of size (1, 768, 768) and have been thresholded.

- dataset/
  - sample_train/
    - extrapolation/
      - octane-fixed/
        - 1
        - 2
        - ... (288 folders, each containing 5 images)
    - interpolation/
      - octane-fixed/
        - 1
        - 2
        - ... (288 folders, each containing 3 images)
  - sample_test/
    - extrapolation/
      - octane-fixed/
        - 1
        - 2
        - ... (72 folders, each containing 5 images)
    - interpolation/
      - AEC-test-set-threshold
      - ACB-test-set-threshold
      - CED-test-set-threshold (each of these three folders contains 72 subfolders with 3 images)

| **Item**               | **Value**                      |
|------------------------|--------------------------------|
| **Hyperparameters**    |                                |
| Batch Size             | 72                             |
| Learning Rate (lr)     | 0.001                          |
| Epoch                  | 500                            |
| **Dataset**            |                                |
| All Samples            | 4320                           |
| Training Set           | 80% (3456 in total)            |
| Test Set               | 20% (864 in total)             |
|                        | **Interpolation** | **Extrapolation** |
| Training Sequences     | 288                            | 288 |
| Test Sequences         | 72                             | 72  |
| Images in a Sequence   | 3                              | 5   |


### Training
Training source code is arranged using Python:
```
python ./LSTM_training.py
```

### Prediction
Predicting source code is arranged using Python:
```
python ./predict.py

###

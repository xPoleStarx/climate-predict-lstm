# Climate Prediction Using LSTM

This project aims to predict the mean temperature in Delhi using an LSTM (Long Short-Term Memory) neural network. The data includes daily climate observations such as temperature, humidity, wind speed, and pressure.

## Table of Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model](#model)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [License](#license)

## Dataset
The dataset used in this project is split into two files:
- `DailyDelhiClimateTrain.csv`
- `DailyDelhiClimateTest.csv`

Both files contain the following columns:
- `meantemp`: The mean temperature
- `humidity`: The humidity level
- `wind_speed`: The wind speed
- `meanpressure`: The mean pressure

## Requirements
To run this project, you need the following dependencies:
- Python 3.7+
- NumPy
- Pandas
- PyTorch
- Scikit-learn

You can install the dependencies using `pip`:
```bash
pip install numpy pandas torch scikit-learn
```
## Model
The LSTM model is defined in main.ipynb. The key parameters are:
```bash
input_size: Number of input features (4 in this case: meantemp, humidity, wind_speed, meanpressure)
hidden_size: Number of hidden units in the LSTM layer (64)
num_layers: Number of LSTM layers (2)
output_size: Number of output features (1, since we are predicting meantemp)
```
## Training and Evaluation
The training process involves the following steps:

Reading and preprocessing the data: The data is read from CSV files and converted to NumPy arrays.
`Creating sequences:` The data is transformed into sequences to fit the LSTM model.
`Creating DataLoaders:` PyTorch DataLoaders are created for batching the data.
`Defining the model:` An LSTM model is defined using PyTorch.
`Training the model:` The model is trained using Mean Squared Error (MSE) loss and Adam optimizer.
`Evaluating the model:` The model is evaluated on the test set to compute the test loss, Mean Squared Error (MSE), and R² score.
## Results
After training, the model achieved the following performance on the test set:

Test Loss: 4.4735
Mean Squared Error: 4.7441
R² Score: 0.8852
Here are the first five actual vs. predicted temperatures:
```python
Actual: [15.684211, 14.571428, 12.111111, 11.0, 11.789474]
Predicted: [28.027569, 15.272582, 14.947316, 14.326501, 13.310998]
```

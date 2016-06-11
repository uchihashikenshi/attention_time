# Attention Neural Network for Time-Series
AttentionalTime is a Python implementation of a time-series model with (optional) attention where the encoder is CNN, decoder is LSTM. 

The attention model is from __.

This project is maintained by Kenshi Uchihashi.

## Dependencies
### Python

* numpy
* tensorflow

## Quickstart
We are going to working with some example data in data/ folder. It contains delicious dataset, Twitter hashtags, and so on. First run the data-processing code like this:

```
python preprocessing.py
```

## Details
### Preprocessing options(```preprocessing.py```)
#### Data options 

* ```data_dir```: 

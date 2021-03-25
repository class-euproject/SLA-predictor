# SLA Predictor

-----------------------

### Acknowledgements

This work has been supported by the EU H2020 project CLASS, contract #780622.

-----------------------

### Microservice

1. Build and run microservice image from microservice folder
```
docker build -t micro-sla .

docker run -d --network-alias=micro-sla --name micro-sla -p 5002:5002 micro-sla
```

2. Call the predictSLA function from the microservice with values for workers (X) and exectime (Y)
```
localhost:5002/predictSLA?workers=X&exectime=Y
```

### Notebooks

```
CLASS_SLA-EDA.ipynb -> Exploratory Data Analysis
CLASS_SLA-data.ipynb -> Training data generation
CLASS_SLA-model-NN.ipynb -> Neural networks research
CLASS_SLA-model-regression.ipynb -> Regression models benchmark
CLASS_SLA-model-classification.ipynb -> Classification models benchmark and final model implementation
```

More information about the notebooks is detailed in "D4.6 Validation of the Cloud Data Analytics Service Management and Scalability components" in Section 4.2

---------------------------------

### LICENSE

`SLA Predictor` is licensed under [Apache License, version 2](LICENSE).

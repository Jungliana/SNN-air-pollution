Predicting Air Pollution With the Use of Spiking Neural Networks
==============================

Excessive levels of air pollution are one of the biggest 
threats to health of humans and natural environment. Thus, 
it is particularly important to accurately forecast air 
pollution in order to minimize its hazardous effects and 
support multiple day-to-day decision making in various systems. 
As part of this work, the existing state-of-the-art for 
predicting the level of air pollution were reviewed. 
Subsequently, theoretical foundations of Spiking Neural 
Networks (SNNs), which were used to conduct the study, 
are also presented. The author of the thesis proposed and 
implemented three architectures of SNNs based on recently 
introduced snnTorch package. In order to test the prepared 
implementations, the real dataset of Particulate Matter 2.5 
(PM2.5) air pollution for Warsaw was selected and preprocessed. 
As the results of the experiments showed, the selected spiking 
neural networks performed better than the non-spiking artificial 
multilayer perceptron.

Project Organization
------------

    ├── README.md
    ├── data
    │   ├── processed     <- Final, processed data in csv files.
    │   └── raw           <- Raw data in text files.
    │
    ├── notebooks         <- Jupyter notebooks.
    │   ├── data_processing.ipynb
    │   ├── dataset_properties.ipynb
    │   ├── profiling.ipynb
    │   └── training_stats.ipynb
    │
    ├── src      <- Source code for use in this project.
    │   ├── __init__.py
    │   ├── error_measures.py   
    │   ├── graphs.py   
    │   ├── models.py       
    │   └── train_model.py         
    │
    └── requirements.txt

--------

# Prediction of New Daily COVID-19 Cases in Malaysia

This Deep Learning Time Series model is an assigment project to predict the new daily COVID-19 cases in Malaysia.

## Table of Contents
* [General Info](#general-information)
* [Technologies](#technologies)
* [Installation](#installation)
* [Model Development](#model-development)
* [Credits and Acknowledgments](#credits-and-acknowledgements)
* [Contact](#contact)

## General Information

This Deep Learning Time Series model is able to predict the new daily COVID-19 cases in Malaysia based on the past 30 days number of cases. It will help scientists and policy makers alike in making the best decision to curb the wide spread of the virus such as travel ban.

The year 2020 was a catastrophic year for humanity. Pneumonia of unknown aetiology was first reported in December 2019., since then, COVID-19 spread to the whole world and became a global pandemic. More than 200 countries were affected due to pandemic and many countries were trying to save precious lives of their people by imposing travel restrictions, quarantines, social distances, event postponements and lockdowns to prevent the spread of the virus. However, due to lackadaisical attitude, efforts attempted by the governments were jeopardised, thus, predisposing to the wide spread of virus and lost of lives.

## Technologies

- Python <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="drawing" width="6%"/>
    - Spyder (Python 3.8.13) <img src="https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon" alt="drawing" width="6%"/>
    - Google Colab (Python 3.6) <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Google_Colaboratory_SVG_Logo.svg/1200px-Google_Colaboratory_SVG_Logo.svg.png" alt="drawing" width="3.5%"/>
        - Pandas 1.4.3 <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="drawing" width="6%"/>
        - Numpy 1.22.3 <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="drawing" width="6%"/>
        - Matplotlib 3.5.1 <img src="https://matplotlib.org/_static/images/logo2.svg" alt="drawing" width="6%"/>
        - Scikit Learn 1.0.2 <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="drawing" width="6%"/>
        - Tensorflow 2.3.0 <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="drawing" width="6%"/>
 
## Installation

1) Go to [Google Colab](https://colab.research.google.com/) and sign in with your Google account
2) Download the models folder from the repository and upload it in Google Colab
3) Run the model

## Model Development

- This dataset is a Time Series problem, thus a Deep Learning Long Short Term Memory (LTSM) algorithm was used.
- The following is the architecture of the LTSM model of 4 layers that was used:

![model](statics/model.png)

From the algorithm above, the model was able to produce prediction with Mean Absolute Percetage Error (MAPE) of 9.34%.

Below is the computed MAPE score:

![Model Evaluation](statics/model_evaluation.png)

Using Tensorboard, the model training was visualized as follows:

![Tensorboard Epoch Loss](statics/tensorboard_loss.png)

![Tensorboard Epoch Acc](statics/tensorboard_mape.png)

From the visual above, we can see that the model's training loss decreased significantly around 20th epoch and stabilize around 40th epoch, before it fluctuated again from 110th epoch above. Using Model Checkpoint callback in the training of the model helped us to identify the best model within all the epoch iterations.

Finally, we visualized the predicted new daily COVID-19 cases and the actual daily COVID-19 cases below:

![Time Series Actual Predicted](statics/time_series_actual_predicted.png)

From the graph above, the model was able to predict very closely in the first 70 days, before it lags from there on when the new daily cases suddenly burst. However, we can observe that the model was still able to predict the trend of the new daily cases, which is still useful in giving insights to the scientists and policy makers.

## Project Status

Project is completed for the assignment.

## Credits and Acknowledgements

This data was sourced from [Ministry of Health Malaysia | GitHub](https://github.com/MoH-Malaysia/covid19-public).

Special thanks to Alex Koh and Warren Loo from SHRDC and Dr Mohammed Al-Obaydee from HRD Academy for the guidance and training to make 
this project possible.

## Contact

Created by [@Muhammad Al Mubarak Zainal Abeeden](https://www.linkedin.com/in/m-almubarak-za/) - Any suggestions or feedbacks are welcomed. Feel free to contact me!

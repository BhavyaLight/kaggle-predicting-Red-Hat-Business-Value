[![Stories in Ready](https://badge.waffle.io/BhavyaLight/kaggle-predicting-Red-Hat-Business-Value.png?label=ready&title=Ready)](https://waffle.io/BhavyaLight/kaggle-predicting-Red-Hat-Business-Value)
# kaggle-predicting-Red-Hat-Business-Value

## Development set up

### Requirements
1. Docker
#### Installation steps:
Pull the image from docker hub  
`Docker pull bhavya2107/machinelearning`  

Or navigate to the current project directory, make sure the requirements.txt exists and   
`Docker build .`  
This step will take some time.  

#### Run the docker container
Navigate into the github directory for this project and run:   
`docker run -it -v $(pwd):/src bhavya2107/machinelearning`  

## Pre-processing steps:
Once inside the docker container, navigate into the pre-processing folder. 
Run code for label encoding:  
`python preprocessing_label_encoding.py --data_directory <file-path-to-Data-directory>`  
*Note*: Before running the above code, download the data from Kaggle and store it in a directory called 'Data'. Extract the files and do not change the names.

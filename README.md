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
1. Run code for label encoding:  
`python preprocessing_label_encoding.py --data_directory <file-path-to-Data-directory>`  
*Note*: Before running the above code, download the data from Kaggle and store it in a directory called 'Data'. Extract the files and do not change the names.
2. Run code for reducing redundant feature for performance improvement:  
`python preprocessing_elimate_redundancy.py --data_directory <file-path-to-Data-directory>`  
*Note*: Before running the above code, label encoded files from step 1 must be present in your data directory. Do not change their names.  
ERROR HACK: A slight bug, which has not been automatically corrected and needs to be done manually as of now.  
After step 2, open the `act_test_features_reduced.csv` and insert a row. Copy and paste any row, change the activity id to `act_0` and change the cell values for _people_char_3, char_1, char_2, char_5_ columns to '99999'.

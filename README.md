[![Stories in Ready](https://badge.waffle.io/BhavyaLight/kaggle-predicting-Red-Hat-Business-Value.png?label=ready&title=Ready)](https://waffle.io/BhavyaLight/kaggle-predicting-Red-Hat-Business-Value)
# kaggle-predicting-Red-Hat-Business-Value

## Development set up

### Requirements
1. Anaconda installation with multithreading xgb (local set up)
2. xgboost installation is not supported by anaconda or pip - so needs to be installed manually on your local drive with instructions from their docs (However, the documentation is slightly tricky)
`https://github.com/dmlc/xgboost/blob/master/doc/build.md`

## Pre-processing steps:
0. Create a 'Data' folder. Store the unzipped data files, also create a 'pickle' folder inside the data folder
1. Run code for label encoding:  
`python preprocessing_label_encoding.py --data_directory <file-path-to-Data-directory>`  
*Note*: Before running the above code, download the data from Kaggle and store it in a directory called 'Data'. Extract the files and do not change the names.
2. Run DimensionalityReduction_with17304_removal.ipynb
3. Run PreprocessingInterpolation.ipynb
4. Run Preprocessing_dateFeatures.ipynb
5. Run group_outcome_change.ipynb
6. Run Preprocessing_merging.ipynb.
** Note: _ Preprocessing_merging.ipynb required you to check if all categorical variable are one-hot-encodable i.e. there are no inconsistency in the total unique value in a OHE column in test and train. An additional row maybe added to make it consistent. See comment block in the file **
7. Run xgboost.ipynb

_Additional Note:_ Depending on the path to your data folder, you may need to change file paths in the ipynb. These are always present at the beginning of each notebook.

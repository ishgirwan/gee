# Predicting village level HDI by applying Machine Learning to Satellite data

## Data

### Ground truth data

The village level HDI data was published by the state of Karnataka in 2015. The [data](http://www.sirdmysore.gov.in/GPHDI/GPandVillageHDIReport.pdf) is from 27,075 villages. The shapefile data of villages available at [Karnataka's GIS](https://kgis.ksrsac.in/kgis/) has an unique identifier in the form of census codes. But the HDI dataset doesn't have an unique identifier, so data from Indian government's [local government directory](https://lgdirectory.gov.in/) was used to map it to census codes. Fuzzy logic algorithm was used to execute this mapping.The data was cleaned and mapped to relevant village geometry by using census code as the unique identifier. The final cleaned ground truth consists of data from 18,984 villages.

|![HDI values of 27,075 villages](https://github.com/ishgirwan/hdi_prediction/blob/master/readme_images/HDI%20of%2027%2C075%20villages.png) | ![HDI values of 18,984 villages](https://github.com/ishgirwan/hdi_prediction/blob/master/readme_images/HDI%20of%2018%2C984%20villages.png)
|:---:|:---:|
| HDI values of 27,075 villages|HDI values of 18,984 villages |

### Satellite data

The satellite data used for model training was from Landsat 8 Surface Reflectance Tier 1. Following features were used to filter the images:
* Multispectral bands: RED, GREEN, BLUE, NIR, SWIR1 and SWIR2 
* Resolution: 30 meters
* Date: 2015-01-01 - 2016-12-31
* Cloud cover: < 1%
* Reducer: Median 
* Number of Pixels per band ranged from 5 to 65,720

[Google Earth Engine's](https://earthengine.google.com/) Python API was used to download the images. The process took almost 5 days to download the complete image data.

|![Village name:Goudur](https://github.com/ishgirwan/hdi_prediction/blob/master/readme_images/village.png) | ![Number of Pixels](https://github.com/ishgirwan/hdi_prediction/blob/master/readme_images/number%20of%20pixels.png)
|:---:|:---:|
| Sample - Village name: Goudur| Range of number of Pixels per band in the dataset|

### Download dataset

The dataset is hosted at [Kaggle](https://www.kaggle.com/ishgirwan/predict-hdi-of-villages-using-satellite-imagery) and is publicly available.

## Modelling
### Algorithm
Algorithm details:
* 90% of data was used for training while 5% each for validation and test set.
* Architecture - Convolutional Neural Network 
* Model: ResNet18
  * 18 Convolutional layer
  * 11 Million parameters
  * The number of input channels in the first convolutional layer was changed to number of bands.
  * A Sigmoid layer was added in the final layer
* Loss: Mean Square Error(MSE)
* Optimizer: Adam
* Learning rate scheduler: ReduceLRonPlateau
* Batch size: 16
* Image size: The image sizes are of different shapes but the model needs them in the same size in each batch. So, a custom `collate_fn` was written to pad the images with zeros to make them of the same shape. 

## Learnings
### Training results
Training was stopped after 20 epochs as the model’s training loss and validation loss stopped decreasing. The best model produced a MSE of 0.01168 on the validation set and 0.010587 on the test set. A larger model may perform better as the data seems to have under fitted on ResNet18. 

### Analysis
The predicted values seems to be biased towards the mean of the ground truth values. Regularization may help in reducing this bias.

|![HDI values of test set](https://github.com/ishgirwan/hdi_prediction/blob/master/readme_images/test%20set%20values.png) | ![Predicted HDI values on test set](https://github.com/ishgirwan/hdi_prediction/blob/master/readme_images/Prediction%20%20on%20test%20set.png)
|:---:|:---:|
| HDI values of test set| Predicted HDI values on test set|


For 50% of the predictions, error is less than 0.067347.
|![Prediction error](https://github.com/ishgirwan/hdi_prediction/blob/master/readme_images/prediction%20error.png) |
|:---:|
| Prediction error|

###### Example of two villages with similar number of pixels but very distinct prediction error.

|![high error example](https://github.com/ishgirwan/hdi_prediction/blob/master/readme_images/high%20error%20image.png) | ![low error example](https://github.com/ishgirwan/hdi_prediction/blob/master/readme_images/low%20error%20image.png)|
|:---:|:---:|
| Village: Kuveshi(No. of Pixels: 65,053 Prediction error: 0.336)| Village: Mandalekanahalli(No. of Pixels: 65,720 Prediction error: 0.0003) |

### Challenges
* Poor resolution of publicly available Satellite data makes it difficult for the model to learn.
* The ground truth data had several issues:
  * More than 7,000 new villages were created in Karnataka between 2015 and 2019. The HDI data is from 2015 but the village boundaries are updated as new villages are created. This temporal discrepancy would add noise to the data.
  * Several villages with same names
  * Spelling mistakes in village names
  * Lack of an unique identifier
* Deep learning models require GPUs for faster computation. Google Colaboratory offers a free GPU service but as these are dynamically allocated, they have a limited memory and are not very fast. This limits quicker experimentations and an inability to use larger architectures. 

## Dependencies
* PyTorch
* PyTorch lightning
* NumPy
* Pandas
* Google Earth Engine
* GeoPandas
* GEEMap
* Rasterio

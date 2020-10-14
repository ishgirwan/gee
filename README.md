# Predicting village level HDI by applying Machine Learning to Satellite data

## Data

### Ground truth data

The village level HDI data published by the state of Karnataka in 2015. The [data](http://www.sirdmysore.gov.in/GPHDI/GPandVillageHDIReport.pdf) is from 27,075 villages. The data was cleaned and mapped to relevant village geometry  by using census code as the unique identifier. The shapefile data of villages available at [Karnataka's GIS](https://kgis.ksrsac.in/kgis/) has a unique identifier in the form of census codes. But the HDI dataset doesn't have a unique identifier, so data from Indian government's [local government directory](https://lgdirectory.gov.in/) was used to map to census codes. Fuzzy logic algorithm was used to execute this mapping.The final cleaned ground truth consists of data from 18,984 villages.

### Satellite data

The satellite data used for model training was from Landsat 8 Surface Reflectance Tier 1. Following features were used to filter these images:
* Multispectral bands: RED, GREEN, BLUE, NIR, SWIR1 and  SWIR2 
* Resolution: 30 meters
* Date: 2015-01-01-2016-12-31
* Cloud cover: <1%
* Reducer: Median 
* Number of Pixels per band ranged from 5 to 65,720

[Google Earth Engine](https://earthengine.google.com/) with Python API was used to download the images. The process took almost 5 days to download the complete image data.

## Modelling
### Algorithm
Algorithm details:
* 90% of data was used for training while 5% each for validation and test set.
* Architecture - Convolutional Neural Network 
* Model: ResNet18
  * 18 Convolutional layer
  * 11 Million parameters
  * The number of input channels in the first conv layer was changed to number of bands.
  * A sigmoid layer was added in the final layer
* Loss: Mean Square Error(MSE)
* Optimizer: Adam
* Learning rate scheduler: ReduceLRonPlateau
* Batch size: 16
* Image size: Since the image sizes are of different shapes but the model needs them in the same size in each batch. So, a custom `collate_fn` was written to pad the images with zeros to make them of the same shape. 

## Learnings
### Training results
Training was stopped after 20 epochs as the model’s training loss and validation loss stopped decreasing. The best model produced a MSE of 0.01168 on the validation set and 0.010587 on the test set. A larger model may perform better as the data seems to have under fitted on ResNet18. 

### Analysis
The predicted values seems to be biased towards the mean of the ground truth values. Regularization may help in reducing this bias.
For 50% of the predictions, error is less than 0.067347. 

### Challenges
* Poor resolution of publicly available Satellite data makes it difficult for the model to learn.
* The ground truth data had several issues:
  * More than 7,000 new villages were created in Karnataka between 2015 and 2019. The HDI data is from 2015 but the village boundaries are updated as new villages are created. This temporal discrepancy would add noise to the data.
  * Several villages with same names
  * Spelling mistakes in village names
  *  Lack of an unique identifier
  *  Deep learning models require GPUs for faster computation.Google Colaboratory offers a free GPU service but these are dynamically allocated, have a limited memory and are not very fast. This limits quicker experimentations and an inability to use larger architectures. 

## Dependencies
* PyTorch
* PyTorch lightning
* NumPy
* Pandas
* Google Earth Engine
* GeoPandas
* GEEMap
* Rasterio

# Identify

An API that does image predictions, and explains what parts of the image contributed to the assessment.

## Basic Specifications
- Using an inception_v3 network pre-trained on the Imagenet database
- Demo of visualising the superpixels that lead towards the specific predictions

## Sample Results
  ![Image of dognoise](/src/uploads/funnydog.jpeg)
  ![Image of dognoise](/src/uploads/processedfunnydog.jpeg)


## Requirements
- Flask
- Lime
- Torch
- Numpy
- SkImage

## How to run
  '''
  cd src/
  export FLASK_APP = main.py
  flask run
  '''

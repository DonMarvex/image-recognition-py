from imageai.Prediction import ImagePrediction
import os
execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5" ))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "house.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
     print(eachPrediction , " : " , eachProbability)



# The Model here is pretty sure that it is a boathouse by about 93%.
# boathouse  :  92.79947876930237
# beacon  :  7.10703432559967
# dock  :  0.011492978956084698
# church  :  0.01087701748474501
# breakwater  :  0.006240638322196901


'''
ImageAI
    ImageAI, an open source python library built to empower developers to build applications and systems  with self-contained Computer Vision capabilities",
    "https://github.com/OlafenwaMoses/ImageAI",
'''
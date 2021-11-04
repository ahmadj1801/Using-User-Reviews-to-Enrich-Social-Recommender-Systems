# RevNet: Using-User-Reviews-to-Enrich-Social-Recommender-Systems
This is an implementation which seeks to utilise user reviews in a social recommender for ratings prediction.

# Abstract
Recommender systems have become increasingly popular in the online domain and play a critical role in suggesting information of interest to users. Various techniques have been explored while implementing these systems, such as collaborative filtering, content-based and preference-based solutions. In recent times, personal data has become ubiquitous and has prompted researchers to explore avenues that use user reviews on items of interest and social relationships to improve output recommendations. Furthermore, to the foregoing, the immense interest in Neural Networks has provided a platform for applying deep learning techniques to improve existing recommender system solutions. In this paper, we present a novel deep neural network framework (RevNet) for social recommendations through the utilization of user reviews. The model captures the user-user and user-item spaces typically found in a social recommender, as well as adds another dimension in the form of a user-review space. Experiments on the Yelp dataset is done in order to measure the proposed models effectiveness.

# Introduction


# Proposed Model: RevNet


# Code
The code is structured in a way that allows for model hyper parameter variation. To run the model with a default set of parameters, execute the following command into your machines terminal
* `python application.py`
To run the model with custom hyper paramaters, the following format should be adhered to
* `python application.py --parameter1 <value>`
* `python application.py --epochs 50 --batch_size 128 --d2v_vec_size 256`

# Environment Settings
* `Python`: 3.9
* `Keras`: 
* `Gensim`: 

* Other Libraries needed:
  * `Argparse`
  * `Matplotlib`
  * `NLTK`
  * `Numpy`
  * `Pandas`
  * `Pydot`
  * `Sci-kit Learn`
  * `Tensorflow`

# Dataset
Ensure that the dataset files are stored in the `Data` directory. 
* Users: https://drive.google.com/file/d/1P2EIB6fJFqSXL6R3NQ1HW6uO9tL_VCHO/view?usp=sharing
* Businesses: https://drive.google.com/file/d/1Ccg_4U4Md_bjUE8AtaWbzGOaldGZ0v0b/view?usp=sharing
* Reviews: https://drive.google.com/file/d/13SNDU5v3d3TDNluINVTeKDcP1VXQA3dm/view?usp=sharing

* `Alternatively, you can download the complete Yelp dataset here:` https://www.yelp.com/dataset/download

# Running the Code

# Acknoledgements

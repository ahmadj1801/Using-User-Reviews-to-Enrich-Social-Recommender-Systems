# RevNet: Using-User-Reviews-to-Enrich-Social-Recommender-Systems
This is an implementation which seeks to utilise user reviews in a social recommender for ratings prediction.

## Abstract
Recommender systems have become increasingly popular in the online domain and play a critical role in suggesting information of interest to users. Various techniques have been explored while implementing these systems, such as collaborative filtering, content-based and preference-based solutions. In recent times, personal data has become ubiquitous and has prompted researchers to explore avenues that use user reviews on items of interest and social relationships to improve output recommendations. Furthermore, to the foregoing, the immense interest in Neural Networks has provided a platform for applying deep learning techniques to improve existing recommender system solutions. In this paper, we present a novel deep neural network framework (RevNet) for social recommendations through the utilization of user reviews. The model captures the user-user and user-item spaces typically found in a social recommender, as well as adds another dimension in the form of a user-review space. Experimentation on a real-world dataset, the Yealp dataset, is done, in order to measure the effectiveness of the model.

## Introduction
Recommender Systems have been around for quite some time. These systems provide personalised service support to users by learning their previous behaviours and predicting their current preferences [1]. Moreover, there has been a rapid increase in web technologies that employ recommendation systems to enhance the users' experience. The driving factor behind this is that it solves a major problem surrounding today's times, information overload. Recommender systems can be used to overcome this phenomenon by automating the decision-making process and providing a set of informed recommendations, thereby enhancing a user's experience [1]. The incorporation of social relations into recommender systems has been backed by social theories, where it is argued that people are influenced by their social connections, which ultimately leads to them having similar preferences [2] [3] [4]. 


As depicted in figure below, a user is involved in both the item and social spaces of a social recommender system. The user-item space denotes the interactions between a user and items, whereas the user-user space captures the friendships amongst users.

![user-item-user-fig](https://user-images.githubusercontent.com/24585616/140293799-dc0f602f-d083-4b19-82fd-edd42b88544b.png)


Furthermore, the aim of our work is to utilise user reviews in a social recommender system. Thus, we propose to add a user-review space to learn item latent factors. As shown in the figure below, an item may have multipl reviews from many users.


< Insert figure here >

## Proposed Model: RevNet


## Environment Settings
* `Python`: 3.9
* `Keras`: 
* `Gensim`: 

### Other Libraries needed:
  * `Argparse`
  * `Matplotlib`
  * `NLTK`
  * `Numpy`
  * `Pandas`
  * `Pydot`
  * `Sci-kit Learn`
  * `Tensorflow`

## Dataset
Ensure that the dataset files are stored in the `Data` directory. 
* Users: https://drive.google.com/file/d/1P2EIB6fJFqSXL6R3NQ1HW6uO9tL_VCHO/view?usp=sharing
* Businesses: https://drive.google.com/file/d/1Ccg_4U4Md_bjUE8AtaWbzGOaldGZ0v0b/view?usp=sharing
* Reviews: https://drive.google.com/file/d/13SNDU5v3d3TDNluINVTeKDcP1VXQA3dm/view?usp=sharing

* `Alternatively, you can download the complete Yelp dataset here:` https://www.yelp.com/dataset/download

## Running the Code
The code is structured in a way that allows for model hyper parameter variation. To run the model with a default set of parameters, execute the following command into your machines terminal
* `python application.py`


To run the model with custom hyper paramaters, the following format should be adhered to
* `python application.py --parameter1 <value>`
* `python application.py --epochs 50 --batch_size 128 --d2v_vec_size 256`

## Acknowledgements

## References
* [1] Q. Zhang, J. Lu, and Y. Jin, “Artificial intelligence in recommender systems,” Complex & Intelligent Systems, vol. 7, no. 1, pp. 439–457, Nov. 2020, doi: 10.1007/s40747-020-00212-w.
* [2] W. Fan, Y. Ma, D. Yin, J. Wang, J. Tang, and Q. Li, “Deep Social Collaborative Filtering,” doi: 10.1145/3298689.3347011.
* [3] Eytan Bakshy, Itamar Rosenn, Cameron Marlow, and Lada Adamic. 2012. The role of social networks in information diffusion. In World Wide Web conference. 519–528.
* [4] Wenqi Fan, Tyler Derr, Yao Ma, Jianping Wang, Jiliang Tang, and Qing Li. 2019. Deep Adversarial Social Recommendation. In Proceedings of the 28th International Joint Conference on Artificial Intelligence, IJCAI-19.



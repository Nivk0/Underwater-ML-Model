# Clear Image Translation with CycleGAN

***Faculty Advisor:*** [Dr. Yapeng Tian](https://www.linkedin.com/in/yapeng-tian-780795141/) <br>
***Team Lead:*** [Ben Bowers](https://www.linkedin.com/in/benhbowers/) <br>
***Participants:*** [Murtaza Khan](https://www.linkedin.com/public-profile/settings?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_self_edit_contact-info%3Bv4eX%2B99wQFCEphySFPZ4KQ%3D%3D), Ryan Joseph, Saanvi Bala, Nivedh Koya, Ompranay Yedluri 

![poster](https://github.com/ACM-Research/Image-Transfer/blob/main/poster/poster.png?raw=true)

# Introduction
The <ins>**CycleGAN (Cycle Generative Adversarial Network)**</ins> is a technique that involves the automatic training of image-to-image translation models. It is an approach to training deep convolutional neural networks. The function of this network is to learn mapping between input and output images using a given set of datasets. This simple technique is powerful, achieving visually impressive results on a range of application domains. We have applied the same technique to build a model that can translate underwater cloudy images to clear images and thus enable efficient object detection in deep sea waters.

# Model

- <ins>**Semi-supervised machine learning**</ins> is a combination of supervised and unsupervised learning. It uses a small amount of labeled data and a large amount of unlabeled data, which provides the benefits of both unsupervised and supervised learning while avoiding the challenges of finding a large amount of labeled data.

- <ins>**An Unsupervised Model**</ins> is useful for training as it helps produce accurate translated images and allows our model to quickly detect patterns from a given unlabeled dataset. This allows it to efficiently recognize various image classes. Moreover, an unsupervised model allows the entire model to swiftly adapt to the newer, unlabeled data being inputted that can account for high variance. 

- <ins>**A Supervised Model**</ins> is used to define a set of target classes for the modelto run off of when it’s given diverse, unlabeled data. The unsupervised model consequently leans on the supervised model to provide an accurate result image. 

# Image Pre-Processing
To have our model properly parse our images through, we must first process the images in a way that the code can understand. After importing our images from their folders, they are organized into NumPy arrays to be iterated through. When we append the images, they are resized into a 128x128 size to create consistency throughout the array. Next, our image arrays are then split further between our paired and unpaired images, checking the filenames for the label indicating which category the image is. The <ins>**reason for separating the two types is to allow our Semi-Supervised Model to first be trained on paired data**</ins>, providing a more accurate output. Finally, each of the image arrays are converted into tensors, which can be processed by the models.

# Data Collection
- <ins>**Training the Unsupervised Model**</ins>: To obtain our unpaired (or unlabeled) data, we utilized the EVUP Dataset[^1] that consisted of approximately <ins>3200 underwater images</ins>. These had both cloudy and clear water images which were treated as unpaired. 
- <ins>**Training the Supervised Model**</ins>: For our paired (or labeled) data, we used an underwater camera and took images of objects in clear water and used the same objects to take pictures in cloudy water to replicate the cloudiness of the sea. For better outcomes, we attempted to expand the dataset by using objects of various shapes and sizes (i.e. toy cars, keychains, water bottles, finger rings, etc.). In total, we captured <ins>284 images or 142 pairs of underwater images</ins> to train our supervised model.

<p align="center">
  <img width="450" height="250" src="https://github.com/ACM-Research/Image-Transfer/blob/main/poster/car_clear.jpg"> 
</p>

<p align="center">
  <img width="450" height="250" src="https://github.com/ACM-Research/Image-Transfer/blob/main/poster/car_cloudy.jpg"> 
</p>

# Results

<p align="center">
  <img width="700" height="500" src="https://github.com/ACM-Research/Image-Transfer/blob/main/poster/result_main.png"> 
</p>

<p align="center">
  <img width="700" height="500" src="https://github.com/ACM-Research/Image-Transfer/blob/main/poster/result_second.png"> 
</p>

# Processing Model
<ins>**Adversarial loss**</ins>: This loss is used to train the discriminator network (d) to correctly classify real and fake images. It is a mean absolute error (mae) loss and is used to optimize the discriminator network.

Identity loss: This loss is used to preserve the identity of the input image after being passed through the generator network (g1). It is also a mean absolute error (mae) loss and is used to optimize the generator network (g2) that is responsible for the identity mapping.

Cycle loss (Forward cycle): This loss is used to ensure that the generated image (g2_out) after being passed through the generator network (g2) and then through the inverse generator network (g1) is close to the original input image (input_img). It is also a mean absolute error (mae) loss and is used to optimize the generator network (g1) responsible for the forward cycle.

Cycle loss (Backward cycle): This loss is used to ensure that the generated image (g2_out_id) after being passed through the identity mapping network (g2) and then through the generator network (g1) is close to the original input image (input_id). It is also a mean absolute error (mae) loss and is used to optimize the generator network (g2) responsible for the backward cycle.

Paired loss: This loss is used to encourage the generated image (g1_out) to be as close as possible to the paired image (paired_img) that has some known relationship with the input image. This is also a mean absolute error (mae) loss and is used to optimize the generator network (g1).

![flowchart](https://github.com/ACM-Research/Image-Transfer/blob/main/poster/flowchart.jpeg?raw=true)

CycleGAN, which consists of two generator networks and two discriminator networks. The overall goal of a CycleGAN is to learn a mapping between two domains, for example, between images of horses and zebras.

The discriminator function in the context of GANs (Generative Adversarial Networks) is trained to distinguish between real and fake images. During training, the generator network is updated to generate images that are increasingly difficult for the discriminator to distinguish from real images.

the following steps:

Take a real image from domain A.
Use generator A to B to create a fake image in domain B.
Use generator B to A to create a fake image in domain A.
Compare the fake image in domain A with the real image in domain A using the discriminator A.
Compare the fake image in domain B with the real image in domain B using the discriminator B.
Use the feedback from the discriminators to improve the generator networks.

# Conclusion
With the extensive information we’ve gathered, we find that our model clarifies images. Our results have the input images depicted in a more color accurate, realistic manner, allowing the image to more resemble real life. This model can be utilized for more complex real-world applications with further development. For example, enhancing X-Ray imaging and improving security video camera systems. Diversifying the data set can lead to a more accurate model, thus producing clearer images.

# References
[^1]: The EUVP dataset. (n.d.). Minnesota Interactive Robotics and Vision Laboratory. Retrieved April 25, 2023, from https://irvlab.cs.umn.edu/resources/euvp-dataset

Li, W., Wang, Z., Li, J., Polson, J. S., Speier, W., & Arnold, C. W. (2019). Semi-supervised learning based on generative adversarial network: a comparison between good GAN and bad GAN approach. In Computer Vision and Pattern Recognition (pp. 55–65). https://arxiv.org/pdf/1905.06484.pdf

Yathish, V. (2022b, August 4). Loss Functions and Their Use In Neural Networks. Towards Data Science; Medium. Retrieved March 27, 2023, from https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9


## Face Generation
In this project, we defined and trained a DCGAN on a dataset of faces. Our goal was to get a generator network to generate new 
images of faces that look as realistic as possible!

The project is broken down into a series of tasks from loading in data to defining and training adversarial networks. 
At the end of the notebook, we were able to visualize the results of our trained Generator to see how it performs; our 
generated samples look like realistic faces with small amounts of noise.

### Get the Data
We used the CelebFaces Attributes Dataset (CelebA) to train your adversarial networks.

This dataset is more complex than the number datasets (like MNIST or SVHN) we've been working with, and so, we 
were prepared to define deeper networks and train them for a longer time to get good results. 
Utilize a GPU for training was important.

### Pre-processed Data
Since the project's main focus is on building the GANs, we've done some of the pre-processing for you. Each of the CelebA 
images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. This
data can be downloaded from [here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip)

### Defining the Model
A GAN is comprised of two adversarial networks, a discriminator and a generator.

- Discriminator
Our first task was to define the discriminator. This is a convolutional classifier like we've built before, only without any 
maxpooling layers. To deal with this complex data, it was suggested to use a deep network with normalization. 
  - The inputs to the discriminator are 32x32x3 tensor images
  - The output is a single value that indicates whether a given image is real or fake

- Generator
The generator upsamples an input and generate a new image of the same size as our training data 32x32x3. 
This is mostly transpose convolutional layers with normalization applied to the outputs.
  - The inputs to the generator are vectors of some length z_size
  - The output should be a image of shape 32x32x3
  
### Initialize the weights of your networks
To help our models converge, we initialized the weights of the convolutional and linear layers in your model. From reading the 
[original DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) which says that all weights were initialized from a zero-centered
Normal distribution with standard deviation 0.02.

Following are the main points in weight initialization function
- It initializes only convolutional and linear layers
- It initializes the weights to a normal distribution, centered around 0, with a standard deviation of 0.02.
- The bias terms, if it exist, may be left alone or set to 0.

Next we define our models' hyperparameters and instantiate the discriminator and generator. 

### Discriminator and Generator Losses
Now we need to calculate the losses for both types of adversarial networks.

- Discriminator Losses : For the discriminator, the total loss is the sum of the losses for real and fake images, d_loss = d_real_loss + d_fake_loss.
Remember that we want the discriminator to output 1 for real images and 0 for fake images, so we need to set up the losses to reflect that.
- Generator Loss :The generator loss will look similar only with flipped labels. The generator's goal is to get the discriminator to think its generated images are real.

For both Discriminator and Generator, we used the Adam optimizer with a learning rate of 0.0001. 
Training will involve alternating between training the discriminator and the generator. 
We used our functions real_loss and fake_loss to help us calculate the discriminator losses.

### Implementation
Implementation can be found [here](https://github.com/UsmanIjaz/DL_Face_Generation/blob/master/dlnd_face_generation.ipynb)

### Important takeaways
- The dataset is biased; it is made of "celebrity" faces that are mostly white
- One main thing I am seriously thinking about is some kind of mechanism like early stopping to be used for GANs.
- Upsampling techniques like pixel shuffle can be tried.
- We can have faces from different racial backgrounds, to see how the model performs. Also, we can use labels(racial information) to generate faces for specific racial features.
- I could try a bit more complex discriminator.
Other optimizers and different learning rates could be tried.
  

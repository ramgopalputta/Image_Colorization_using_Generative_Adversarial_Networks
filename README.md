# Abstract of the project:

This study covers the exploration and utilization of Generative Adversarial Networks (GANs) to
colorize grayscale images of various birds common in Western Washington and the Seattle Area. This
work goes over different stages from image augmentation and preparation, GAN model construction,
and various hyper-parameter tuning to help propound the colored image quality result from the
trained network. The study will briefly explore the methodology applied, the theory behind Generative Adversarial Networks, and their utilization to achieve decent models for the selected dataset.
The paper also tries to shortly analyze outcomes and choices taken and the quality of the resulting
models.


# Introduction and Overview:
Bird Watching is a very profound and grossing outdoor activity around the world. It has a significant
impact on stimulating sight and hearing senses by encouraging attentiveness to details, given that there
are thousands of bird species throughout the biosphere. The dataset was selected as bird species tend to
be highly varying in color and therefore would be interesting to attempt to use for this image colorization
project. Realizing that advanced algorithms today can autonomously accomplish arduous tasks like
Image colorization with superb precision and realism is incredibly inspiring. Manually performing image
colorization is complex and time-consuming; even using state-of-the-art graphics editors would still require
tremendous artistic talent from an experienced individual. This study will examine GANs, a reasonably
modern Machine Learning methodology first introduced in 2014 to explore and train a model for coloring
images of produce. The dataset used includes images of 18 different bird species that, for simplicity, are
pre-cropped to ensure that only one bird appears per image and the bird occupies at least 50% of the
pixels in the picture.

# Comutational Results:

The model managed to grasp the basic features of the bird. After the first 80 epochs, the generator
model already outputted vague resemblances of birds. Over the following few hundred epochs, it refined
a few more minor bird details, resulting in increasingly better quality. The results after 400 epochs,
although still a bit blurry looking, got reasonably close to the actual image.

![image](https://user-images.githubusercontent.com/114395443/225207711-94f86561-536e-4de5-9fb2-9270be4dcda9.png)

Although the first model managed to create decent looking images after 400 epochs, it was not good
enough to fool the discriminator, so the loss of the discriminator managed to reach throughout almost
the entire training as shown in the below figure.

![image](https://user-images.githubusercontent.com/114395443/225207871-744d4027-546e-47f3-9dca-de58e9b4d9c3.png)

The first model, although not doing good in regards to the discriminator, also distinguished between bird
species and the surrounding backgrounds of the images like skies, terrains, and tree branches. It managed
to reasonably color both bird species and surroundings correctly. As before, it did
not manage to learn small details, and the results illustrated appear a bit unclear.

![image](https://user-images.githubusercontent.com/114395443/225208035-2815489b-30df-43ae-a74e-41c954232104.png)

The new model, where the generator contained skip connections, did tremendously better over the 50
epochs it was trained, where the data contained augmented images. The images resulted very realistic
starting at epoch 20, it then refined very little colored details from that point forward as shown in the below figure.

![image](https://user-images.githubusercontent.com/114395443/225208151-efc5c541-a6a4-470a-9822-ebaca83a5ef2.png)

From the losses of the second model, it is essential to note that neither the generator nor the discriminator
has prevailed. Since neither the generator and discriminator got very low loss levels(making the other
high), as shown in below figure, it is an indicator that neither model is dominating the other.

![image](https://user-images.githubusercontent.com/114395443/225208278-41c628c2-5680-4893-a90f-673448385ed8.png)

Results from some testing data for the second model show much better performance and clarity of the
colored output. As shown in the below figure, the predicted RGB images look almost indistinguishable from
the authentic images.

![image](https://user-images.githubusercontent.com/114395443/225208399-1a117368-f211-4e4f-80b7-3b733f6c7aba.png)

The second model was also used to predict colored images of random colorful bird species not found in the
training dataset for curiosity, as shown in Figure 12. As expected, the results were not very impressive.

![image](https://user-images.githubusercontent.com/114395443/225208559-a54f97f3-53ce-4c13-aa7d-b57b42991963.png)













# Generative Adversarial Network

NOTE: This was done on Notebooks so each python file was on a separate code block all on one jupyter notebooks

## Purpose
This was an educational project I created for myself to test my mathematical knowledge of dense, convolutions, deconvolutions, maxpooling, padding,
and backpropagating through all these different types of layers. The data I was using was MNIST numbers images.

## GAN specific
I had to spent many hours altaring the number, type and size of layers, eg, I doubled the number of deconvolutional layers in the generator (3 -> 6), and halved the number
of feature maps I was creating for each layer. This was because my GAN was replicating the back ground image, and output noise in the centre of the image, but wasn't getting
the detail so I decided to add more layers. To allow the generator to develop more complex knowledge of creating sharp curves for the numbers.

## General
https://github.com/user-attachments/assets/af6429de-738c-46bb-a932-e5c169eafbab
This is a video the output of every step created from the GEN, I think this is a good visual of way of seeing a GAN, work as
although by the end it cannot create perfect images, you can see how the image evolve from simple circles to advanced characters
with curves, and by end you can see the iamge morphing into different numbers.


This is a graph of the generator loss and discriminator loss, where each marking is after 20 pass-throughs of the GAN.
![image](https://github.com/user-attachments/assets/5dd68439-bd61-4bae-8485-306502ca8a5a)


## Other
The final weights are not really useful due to mode collapse where a GAN will not produce diverse outputs and only produces the same images,
even though the input is random noise.

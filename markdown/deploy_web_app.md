# Deploying a web application for classification using Streamlit

Now that we have trained a **deep learning** model for image classification and saved it, we can integrate this model into a simple **web application** that allows the user to **input** an image and classify it according to our trained model classes. We designed this web app using `Streamlit`, which is a very **user friendly** and simple framework for creating **data-centric** web applications with Python.

The **.py** file for the web app, its **requirements** and the saved **trained model** can be found [here](/streamlit/).
The web application was deployed using **Streamlit Cloud**, and it can be accessed using [this link](https://david1792x-pan-dulce.streamlit.app/).

The app allows the user to choose between **uploading** an image or **taking it** directly with the device camera, if available. 

<div align = 'center'>
  
| <img src='/images/streamlit_1.JPG' height="400">         |
|:-------------------------------------------------:|
| ***Figure 1.**  Streamlit web application*               |
  
</div>

Now, we can input images and make **predictions** on the type of pan dulce contained in the image. First, we download a generic image from the internet, in this case, an image of a **Rosca de Reyes**.

<div align = 'center'>
  
| <img src='/images/streamlit_2.JPG' height="400">         |
|:-------------------------------------------------:|
| ***Figure 2.**  Rosca de Reyes classification with image from the internet*               |
  
</div>

The app **orrectly classifies** the image with a lot of certainty. It also gives the user a **brief description** of the pan dulce type. Now, we can use some **real images** taken from bought pieces of pan dulce to see how the model performs.

<div align = 'center'>
  
| <img src='/images/streamlit_3.JPG' height="400">         |
|:-------------------------------------------------:|
| ***Figure 3.**  Puerquito classification with uploaded image*               |
  
</div>

<div align = 'center'>
  
| <img src='/images/streamlit_4.JPG' height="400">         |
|:-------------------------------------------------:|
| ***Figure 4.**  Barquillo classification with uploaded image*               |
  
</div>

The model correctly identifies the **puerquito** and **barquillo** images with a very high **probability**. Now, lets try with an image taken from afar, containing quite a bit of **noise** (a lot of **background** or **other objects**).

<div align = 'center'>
  
| <img src='/images/streamlit_5.JPG' height="400">         |
|:-------------------------------------------------:|
| ***Figure 5.**  Beso classification with uploaded image*               |
  
</div>

As we can see, the model **correctly** identifies the image as a beso, but the probability is **much lower** due to the noise in the image, since most of the training images were taken from the internet and without a lot of noise. Finally, we can try using a **webcam** to take a photo of a **conchita** and use that as our model input.

<div align = 'center'>
  
| <img src='/images/streamlit_6.JPG' height="400">         |
|:-------------------------------------------------:|
| ***Figure 6.**  Conchita classification with webcam image*               |
  
</div>

The web app **correctly** identifies the **conchita** even if the **quality** of the web cam is not that good, lowering the **probability** a bit. The results are exactly what we expected and the model performs **really well** taking into account the **simplicity** and **objectives** of the project.


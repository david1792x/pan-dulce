# **Extracting dataset using Bing Image Search API**

In any data science project, the most important part is to obtain the data. To do this, we will use the Bing search engine to look for images of pan dulce. Bing is a search engine owned by Microsoft and launched in 2009. It has become a popular option as of lately for its implementation of AI models and ease of use. Here is an example of the search results of pan dulce in Bing.

![Bing search](images/bing.JPG)

We will use Microsoft's Bing Image Search API to scrape image results from the Bing search engine, and download them to create our image dataset. The first step is to import the Python modules needed to make the API calls.


```python
# Import required modules
import hashlib
import os
import requests
import urllib.request
```

Taking the project objectives into account, it was decided to restrict the project to a model that can recognize 10 of the most popular types of pan dulce, and these were chosen based on personal experience and image availability, even though there are a lot more popular types of pan dulce that weren't included. We need to take into account different names for the same type of pan dulce, so we can define a dictionary of some possible names for each type so that we can maximize the data obtained.


```python
# Dictionary with alternative namings for each pan_dulce type
pan_dulce_types = {'conchita': ['conchita', 'concha'],
                    'quequito': ['quequito',
                    'mantecada',
                    'mantecada de chocolate',
                    'quequito de chocolate'],
                    'puerquito': ['puerquito', 'cochinito', 'marranito', 'puerco'],
                    'barquillo': ['barquillo', 'cono de crema'],
                    'orejas': ['orejas', 'orejitas'],
                    'pan-de-muerto': ['pan de muerto'],
                    'rosca-de-reyes': ['rosca de reyes'],
                    'donas': ['donas'],
                    'cuernitos': ['cuernitos', 'cuerno'],
                    'besos': ['besos', 'besitos', 'ojo de buey', 'yoyos']}
```

Microsoft offers the Bing Image Search API as a tool to help users in retrieving data from the web. We set up a resource inside Microsoft Azure cloud platform to handle our API calls, and chose the basic option that allows for up to 1000 calls a month and 3 calls per second. There are higher pricing tiers available for users that need to handle more data, but for our project this will do more than enough. To use the API, we need to get a subscription key from our resource, define the URL and our headers. The subscription key is user inputted for privacy reasons.


```python
# Define API parameters
subscription_key = input('Enter your subscription key: ')
search_url = "https://api.bing.microsoft.com/v7.0/images/search"
headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
```

Now that all of our parameters have been defined, we can move on to doing the actual API calls and downloading our images. First we define a dataset folder that will host subfolders of all of our pan dulce types, and then loop over the alternative namings, performing a query and downloading the images to the specified folder. The number of results will vary with the number of alternate namings to keep the amount of data even between pan dulce types. Finally, we check for duplicate images and remove them, completing the webscraping part of building our image dataset.


```python
#  Download images from Bing
for pan, names in pan_dulce_types.items():
    directory = 'dataset/' + pan
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    queries = [name + ' pan dulce' for name in names]
    for n, query in enumerate(queries):
        nimages = 5 - len(queries)
        for i in range(nimages):
            params  = {'q': query, 'count': 50, 'offset': i * 50}
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()
            thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"]]
            for j, url in enumerate(thumbnail_urls):
                try:
                    urllib.request.urlretrieve(url, directory + '/' + pan + '-'+ str(j + i * 150 + n*300) + ".jpg")
                except:
                    pass
    
    # Remove duplicates
    hashes = set()
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        digest = hashlib.sha1(open(path,'rb').read()).digest()
        if digest not in hashes:
            hashes.add(digest)
            os.rename(path, directory + '/' + filename)
        else:
            os.remove(path)
```

After this, we need to manually check all of our images to remove noise from our dataset. This is a tedious process since pan dulce types tend to have common words in Spanish as names, like conchita meaning shell, beso meaning kiss, and so on. Let's visualize one of the image folders to see its contents.

![Dataset folder](images/puerquito_1.JPG)

At least for the first images downloaded, the results is just what we needed, a bunch of pan dulce images. Now, lets scroll down to the final results and see what we have there.

![Dataset folder](images/puerquito_2.JPG)

We can see that there is quite a bunch of images that do not correspond with our needs, so we remove them manually. Once we have our clean dataset, we reset the name of the images in our dataset and build the .csv file that will be used to store the image path and the class that it corresponds to for our model to process the images.


```python
# Code to rename all images in a directory
import pandas as pd
directory = 'dataset'
df = pd.DataFrame(columns=['filename', 'label'])
labels = {}

for i, folder in enumerate(os.listdir(directory)):
    labels[folder] = i
    if folder != 'test':
        for j, filename in enumerate(os.listdir(directory + '/' + folder)):
            os.rename(directory + '/' + folder + '/' + filename, directory + '/' + folder + '/' + folder + '_' + str(j) + '.jpg')
            df = pd.concat([df, pd.DataFrame({'filename': folder + '/' + folder + '_' + str(j) + '.jpg', 'label': labels[folder]}, index = [0])], axis=0)

# Save dataframe as .csv
df.to_csv('pan_dulce.csv', index = False)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>barquillo/barquillo_0.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>barquillo/barquillo_1.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>barquillo/barquillo_2.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>barquillo/barquillo_3.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>barquillo/barquillo_4.jpg</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



With this done, the only thing left to do is to create the code necessary to import our dataset in a model friendly form. We will use PyTorch to create and train our our deep learning model, so we can define a custom dataset class with a PyTorch friendly format. The dataset class is defined in `pan_dulce_dataset.py`, located [here](dataset/pan_dulce_dataset.py). In the next stage of the project, we will use this dataset to train a convolutional neural network model for classification of pan dulce images.

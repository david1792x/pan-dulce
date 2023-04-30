# Import required modules
import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Load deep learning model
model = torch.load('streamlit\pan_dulce_model.pt')
model.eval()

# Set page style and title
st.set_page_config(layout='wide')
st.markdown('# Pan Dulce Classificator :cupcake:')
st.markdown('## This app classifies images of **Pan Dulce** into 10 different types')
st.markdown("""---""")

# Set columns
empty1, col1, empty2, col2, empty3, col3, empty4 = st.columns([0.5, 1.5, 0.5, 3, 0.5, 3, 0.5], gap = 'small')

# Set session states
if 'upload' not in st.session_state:
    st.session_state.upload = False

if 'camera' not in st.session_state:
    st.session_state.camera = False

# Get user image
torch_img = None
image = None
with col1:
    button_1 = st.button('Upload an image of pan dulce')
    if button_1 or st.session_state.upload:
        st.session_state.upload = True
        with col2:
            if st.session_state.camera:
                st.session_state.camera = False
                st.experimental_rerun()
            image = st.file_uploader('Upload an image of pan dulce', type=['png', 'jpg', 'jpeg'])
            if image is not None:
                image_disp = Image.open(image)
                image_disp = image_disp.resize((400, 400))
                st.image(image_disp, caption='Uploaded image')
                bytes_data = image.getvalue()
                torch_img = torchvision.io.decode_image(torch.frombuffer(bytes_data, dtype=torch.uint8))
            else:
                torch_img = None
                image = None

    st.markdown("""---""")

    button_2 = st.button('Take a picture of pan dulce')
    if button_2 or st.session_state.camera:
        st.session_state.camera = True
        with col2:
            if st.session_state.upload:
                st.session_state.upload = False
                st.experimental_rerun()
            image = st.camera_input('Take a picture of pan dulce')
            if image is not None:
                bytes_data = image.getvalue()
                torch_img = torchvision.io.decode_image(torch.frombuffer(bytes_data, dtype=torch.uint8))
            else:
                torch_img = None
                image = None

# Define image transformations
means = [0.485, 0.456, 0.406]
stdvs = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(means,
                                     stdvs)])

# Predict image category
def predict(torch_img = torch_img):
    if torch_img != None:
        torch_img = transform(torch_img)
        torch_img = torch_img.unsqueeze(0)
        with torch.no_grad():
            probabilities = model(torch_img)
            prediction = torch.max(probabilities, 1)
            probability = prediction[0]
            label = prediction[1]
        return label, probability
    else:
        return None

# Define classes and descriptions
classes = ('Barquillo', 'Beso', 'Conchita', 'Cuernito',
            'Dona', 'Oreja', 'Pan de muerto',
            'Puerquito', 'Mantecada', 'Rosca de Reyes')

description = {0: 'Barquillo, also known as cono de crema, is a type of pan dulce made from a puff pastry cone that is usually filled with custard cream and covered in powdered sugar or an apricot/orange covering',
               1: 'Beso, also known as ojo de buey in some states, is made from two cake-like bread semicircles with jam or cream in the middle. They are usually covered in butter and powdered sugar or jam and coconut shavings',
               2: 'Conchita is the most popular variety of pan dulce. It is shaped like a seashell, and it is made of fluffy bread covered with a sugary glaze of chocolate or vanilla flavor',
               3: 'Cuernito is a versatile type of pan dulce that resembles a croissant. It can be filled with chocolate, vanilla, or strawberry cream, or it can be plain and covered in sugar',
               4: 'Dona is the equivalent of a donut in Mexico. It is a round piece of bread with a hole in the middle, and it is usually covered in sugar, chocolate or other sweet toppings',
               5: 'Oreja is a type of pan dulce made from puff pastry. It is shaped like an ear, hence the name, and it is covered in sugar',
               6: 'Pan de muerto is a type of pan dulce eaten around November 2nd (Dia de los Muertos). It is a round bread with sugar and it is decorated with strips of dough to resemble bones',
               7: 'Puerquito, cochinito or marranito, is a piece of pan dulce shaped like a little piggy. It is made of piloncillo, a type of unrefined sugar or gingerbread, and it is usually covered in sugar',
               8: 'Mantecada, also known as quequito in some places, is the Mexican cousin of a cupcake. It is a small individual cake, usually vanilla or chocolate flavored, covered with an iconic red liner.',
               9: 'Rosca de Reyes is eaten around January 6th (Dia de los Reyes Magos) and it is a eliptical bread with candied fruit or sugar on top. It is filled with a plastic baby Jesus, and whoever gets the baby Jesus in their slice has to buy tamales for everyone on February 2nd (Dia de la Candelaria)'}

# Prediction button 
with col3:
    button_3 = st.button('Classify image')
    st.markdown("""---""")
    if button_3:
        with st.container():
            prediction = predict()
            if prediction != None:
                out = str(classes[prediction[0]]) + ' - ' + str(int(np.exp(prediction[1]) * 100)) + '%'
                st.markdown("<p style='text-align: center; font-size:50px;'>" + out +"</p>", unsafe_allow_html=True)
                st.markdown("""---""")
                st.markdown("<p style='text-align: center; font-size:25px; color: #707070'>" + description[prediction[0].item()] +'</p>', unsafe_allow_html=True)
            
    

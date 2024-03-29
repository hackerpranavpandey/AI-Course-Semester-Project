import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Plant Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT RECOGNITION SYSTEM")
    image_path = "C:\INTERNAL\A\AIProject-main\AIProject-main\home_page.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant names efficiently. Upload an image of a leaf of a plant, and our system will analyze it to detect from which plant this leaf belongs to. Together, let's detect different plants and ensure a fruitful outcome!

    ### How It Works
    1. **Upload Image:** Go to the **Plant Recognition** page and upload an image of a leaf of any plant.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential name of the plant which it belongs to.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Plant Recognition** page in the sidebar to upload an image and experience the power of our Plant Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 9K rgb images of different plant leaves which is categorized into 6 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (6500 images)
                2. test (33 images)
                3. validation (1620 images)

                """)

#Prediction Page
elif(app_mode=="Plant Recognition"):
    st.header("Plant Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Arjun Leaf', 'Curry Leaf', 'Marsh Pennywort Leaf', 'Mint Leaf',
                    'Neem Leaf', 'Rubble Leaf']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

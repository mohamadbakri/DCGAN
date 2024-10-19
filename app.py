import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model



def generate_faces(generator, num_images, randomness_level, width, height):
    generated_images = []

    for _ in range(num_images):
        # Generate random noise
        random_noise = np.random.normal(size=(1, 100))

        # Adjust randomness based on user input
        random_noise *= randomness_level

        # Generate face using the loaded generator
        generated_image = generator(random_noise)[0]

        # Denormalize the image
        generated_image = (generated_image + 1) * 127.5

        # Resize to width x height
        generated_image = tf.image.resize(generated_image, (width, height )).numpy()

        # Convert to PIL Image for display
        pil_image = Image.fromarray(generated_image.astype('uint8'))

        # Append the generated image to the list
        generated_images.append(pil_image)

    return generated_images

def main():
    generator = load_model('generator.h5')
    st.title("Anime Face Generation - DCGAN")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    # st.write("Keras-Tensorflow")
    # st.write("This app generates anime images using a pre-trained DCGAN model.")


    # Loaimport pickleel
    # with open('generator.h5', 'rb') as f:
    #     loaded_generator = pickle.load(f)

    # Number of Images
    num_images = st.slider("Number of Images", 1, 5, 2)
    # Randomness Level
    randomness_level = st.slider("Randomness Level", 0.0, 1.0, 0.5, step=0.1)
    # Image Size
    img_size = st.selectbox('Image size', ['64x64', '128x128', '256x256'])
    width, height = map(int, img_size.split('x'))  # Extract width and height from the string
    

    # Show noise as the starting point
    # starting_point_noise_images = []
    # for _ in range(num_images):
    #     starting_point_noise = np.random.normal(size=(1, 100))
    #     starting_point_noise_image = np.clip((starting_point_noise + 1) * 127.5, 0, 255).astype('uint8')
    #     starting_point_pil_image = Image.fromarray(starting_point_noise_image[0], mode='L')  # Convert to grayscale
    #     resized_noise_image = starting_point_pil_image.resize((width, height))
    #     starting_point_noise_images.append(resized_noise_image)

    # Display the noise images
    # st.write("This is where we started")
    # columns = st.columns(num_images)
    # for i, (img, col) in enumerate(zip(starting_point_noise_images, columns)):
    #     # Display image
    #     col.image(img, caption=f"Noise Image {i + 1}", width=20, use_column_width=True)



    # Generate Faces Button
    if st.button("Generate Faces"):
        generated_images = generate_faces(generator, num_images, randomness_level, width, height)

        # Title for generated images
        st.write("This is where our model takes us")

        # Calculate the number of rows and columns
        num_columns = min(num_images, 5)
        num_rows = (num_images + num_columns - 1) // num_columns

        # Display the generated images in rows and columns
        for row in range(num_rows):
            row_images = generated_images[row * num_columns: (row + 1) * num_columns]
            columns = st.columns(num_columns)

            for i, (img, col) in enumerate(zip(row_images, columns)):
                # Display image
                col.image(img, caption=f"Generated Anime Face {i + 1}", width=20, use_column_width=True)

                # Download button
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                col.download_button(label=f"Download Image {i + 1}", data=img_bytes, file_name=f"generated_image_{i + 1}.png", mime='image/png')

if __name__ == '__main__':
    main()

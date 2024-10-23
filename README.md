
# DCGAN for Anime Face Generation - Streamlit App

![Banner](https://link_to_your_banner_image)

This project demonstrates how to use a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate anime faces, deployed using **Streamlit**. The app allows users to explore the generation of synthetic anime faces from random noise vectors.

## Features

- **DCGAN Model**: Trained to generate 64x64 or 128x128 or 256x256 pixel anime faces.
- **Streamlit Interface**: Simple, interactive web interface for generating new faces.
- **Random Noise Control**: Generate new images based on a random noise vector.
- **Visualization**: View and download generated anime faces.

## Project Overview

### DCGAN Model Architecture

- **Generator**: Takes a random noise vector (latent vector) and outputs a 64x64x3 RGB image. The generator uses transposed convolutions to upsample the latent vector into an image.
- **Discriminator**: A convolutional neural network that distinguishes between real and generated images, helping the generator improve during training.

The model was trained using a dataset of anime faces collected from [Kaggle](https://www.kaggle.com/splcher/animefacedataset).

### Streamlit Deployment

Streamlit is used to create an interactive app, where users can generate new images by interacting with the GAN. The app is easy to use and lightweight, making it perfect for deploying ML models with a simple web interface.

## How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohamadbakri/DCGAN.git
   cd dcgan 
   ```

2. **Install dependencies**:
   Create a virtual environment and install the required packages from the `requirements.txt` file:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501` to interact with the DCGAN model.

## File Structure

```bash
dcgan-streamlit-app/
│
├── app.py                                                                                                                # Streamlit app code
├── generator.h5/                                                                                                     # Pretrained DCGAN generator model
├── Anime Face Generation - DCGAN - Keras-Tensorflow.ipynb/                      # Jupyter notebook
├── requirements.txt                                                                                                # Python dependencies
└── README.md                                                                                                    # This readme file
```

## Model Training

The DCGAN model was trained using the **Anime Face Dataset** from Kaggle. If you'd like to retrain the model:

1. Download the dataset and place it in the `data/` directory.
2. Run the model training script (not provided in this repo, but training scripts should be added for full reproducibility).
3. Save the trained model weights in the `dcgan_model/` folder.

## Example Output

Below are some examples of generated anime faces:

![Example Image 1](https://link_to_example_image_1)
![Example Image 2](https://link_to_example_image_2)

## Requirements

- Python 3.8+
- TensorFlow/Keras
- Streamlit
- NumPy
- Pillow
- tqdm

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Deployment on Streamlit Cloud

To deploy this app on **Streamlit Cloud**, follow these steps:

1. Push your code to a GitHub repository.
2. Go to [Streamlit Cloud](https://share.streamlit.io/), and connect your GitHub repo.
3. Configure the necessary settings (e.g., Python version and dependencies).
4. Click "Deploy"!



## Streamlit Link
https://mohamadbakri-dcgan.streamlit.app/


## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/mohamadbakri/DCGAN.git/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


# 🐾 Animal Image Classifier

An end-to-end **Deep Learning project** that classifies animal images using **Transfer Learning with MobileNet**.  
The project includes **data preprocessing, model training, evaluation, and deployment** as an interactive **Streamlit dashboard**.  

🔗 **Live App:** [Try the Animal Classifier](https://animal-image-classifier-ygetnvrrr6doh2dwrprqfa.streamlit.app/)  

---

## 📖 Project Overview

Image classification is one of the core applications of **computer vision**, and animal classification provides a practical demonstration of deep learning in action.  

In this project, I built and deployed a **MobileNet-based model** that:  
- Leverages **transfer learning** for efficient and accurate classification  
- Learns animal-specific features from a dataset  
- Classifies uploaded images in real time through a Streamlit web app  

This combines **modern deep learning architectures** with a clean deployment pipeline.

---

## ✨ Features

- 📂 Upload any image of an animal  
- 🧠 Model built using **MobileNet (Transfer Learning)**  
- 📊 Real-time predictions with class probabilities  
- 🌐 Lightweight and fast Streamlit dashboard  
- ⚡ Deployed online for easy access  

---

## 🧑‍💻 Tech Stack

- **Python**  
- **TensorFlow / Keras** – MobileNet transfer learning & fine-tuning  
- **NumPy, Pandas, Matplotlib** – Data preprocessing & visualization  
- **Streamlit** – Dashboard & deployment  

---

## 📂 Repository Structure

Animal-Image-Classifier/
- ├── app/ # Streamlit app files
- ├── requirements.txt # Dependencies
- ├── .gitattributes # Git attributes
- ├── dataset/ # Dataset (zipped)
- ├── LICENSE # License file
- ├── README.md # Project documentation
- ├── AnimalClassification.ipynb # Model training & experiments
- ├── .git/ # Git version control
- ├── utils/ # Utility scripts
- ├── mobilenetv2_animals_final.keras # Saved trained model
- ├── myimage.jpg # background image for my dashboard
- └── docs/ # Documentation, assets

  ⚙️ Installation & Setup

- Clone the repository and set up the environment:


git clone https://github.com/Swaransh-Mishra/Animal_Image_Classifier.git
cd Animal_Image_Classifier

- Create a virtual environment:

python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

- Install dependencies:
  pip install -r requirements.txt

▶️ Usage:
- Run the Streamlit App:
  streamlit run app.py

The app will open at: http://localhost:8501

Upload an image → the model will predict the animal category

📊 Model Details

- Architecture: MobileNet (Transfer Learning)

- Why MobileNet?

  - Lightweight and optimized for fast inference

  - Pre-trained on ImageNet → strong feature extraction

  - Ideal for deployment in web apps

- Training: Conducted in Jupyter Notebook (AnimalClassification.ipynb)

- Evaluation Metrics: Accuracy, loss curves, confusion matrix

- Deployment: MobileNet model integrated into Streamlit for real-time predictions

  Future Improvements:

- Extend dataset to cover more animal categories

- Experiment with other pre-trained models (ResNet, EfficientNet)

- Deploy on cloud (AWS/GCP/Azure) for scalability


👨‍💻 Author

Swaransh Mishra

- GitHub: Swaransh-Mishra

- LinkedIn: Swaransh Mishra

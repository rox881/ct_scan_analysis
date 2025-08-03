**🩺 CT Scan Analysis Assistant**

This project is a web-based application built with Streamlit that uses a deep learning model to analyze medical CT scan images. Users can upload a scan, and the application will predict one of several lung conditions (including adenocarcinoma, large cell carcinoma, squamous cell carcinoma) or classify the scan as normal. The analysis results, including confidence scores for each class, are presented on-screen and can be downloaded as a PDF report.

<img width="1366" height="720" alt="image" src="https://github.com/user-attachments/assets/d846ce19-544d-44b5-9626-21000e8fb375" />
<img width="1366" height="720" alt="image" src="https://github.com/user-attachments/assets/8389641d-5048-4567-a35c-eb3d8ed19a53" />



✨ Features
Easy Image Upload: Simple interface to upload CT scan images in various formats (JPG, PNG, BMP).

AI-Powered Analysis: Leverages a pre-trained TensorFlow/Keras model (MobileNetV2) to classify the CT scan image.

Detailed Results: Displays the predicted condition along with a confidence score. It also shows the probability for each potential class.

PDF Report Generation: Users can download a comprehensive analysis report in PDF format for their records.

🛠️ Tech Stack
Application Framework: Streamlit

Machine Learning: TensorFlow

Image Processing: Pillow (PIL)

Numerical Operations: NumPy

PDF Generation: FPDF2

⚙️ Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
Before you begin, ensure you have the following installed on your system:

Python: Version 3.8 or later. You can download it from python.org.

pip: Python's package installer (usually comes with Python).

🚀 Installation Process
Clone the Repository
Open your terminal and run the following command:

git clone https://github.com/your-username/your-repo.git

Alternatively, you can download the app.py file directly.

Navigate to the Project Directory

cd your-repo

Create a Virtual Environment (Recommended)
It's best practice to create a virtual environment to manage project dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install Dependencies
Create a file named requirements.txt in the project directory and add the following lines:

streamlit
numpy
Pillow
fpdf2
tensorflow

Now, install these packages using pip:

pip install -r requirements.txt

Configure the Application
This is the most important step.

Download the Model: You must have the pre-trained model file (mobilenetv2_ct_final.keras). Place it in a known directory (e.g., a models folder within your project).

Update app.py: Open the app.py file and modify the following lines:

Update MODEL_PATH to the correct absolute or relative path of your .keras model file.

CRITICAL: Verify that the CLASS_NAMES list exactly matches the classes your model was trained on, in the correct order.

Run the Application
You're all set! Start the Streamlit server from your terminal:

streamlit run app.py

The application should now be running. Your browser should open a new tab with the app, or you can navigate to the local URL provided in the terminal (usually http://localhost:8501).

📖 Usage
Once the application is running, simply use the web interface:

Click the "Browse files" button to open the file dialog.

Select a CT scan image from your computer.

The image will be displayed on the screen.

Click the "Analyze Scan" button to start the prediction.

The analysis report will appear, and a "Download Report as PDF" button will become available.

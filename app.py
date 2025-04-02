import streamlit as st
import numpy as np
from PIL import Image
import io
from fpdf import FPDF # Using fpdf2 library
import tensorflow as tf

# --- Configuration (MODIFY THESE) ---
MODEL_PATH = r'C:\Users\Gaurav\OneDrive\Desktop\ct\models\mobilenetv2_ct_final.keras'
IMAGE_SIZE = (224, 224)
EXPECTED_CHANNELS = 3

# --- !!! IMPORTANT: UPDATE THIS LIST !!! ---
# Get these names EXACTLY from your Colab training output (train_generator.class_indices)
# Example based on typical CT datasets, VERIFY YOURS:
CLASS_NAMES = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'normal',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]
# --- !!! END OF UPDATE SECTION !!! ---


# --- Model Loading (Cached) ---
@st.cache_resource
def load_my_model(model_path):
    """Loads the pre-trained Keras model."""
    try:
        # Add compile=False if you only need inference and want faster loading
        # or if you run into issues with custom objects not defined at load time.
        # However, since you recompile in Colab, loading normally is fine.
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Error loading model: Model file not found at {model_path}. Please check the path.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.exception(e) # Show full traceback in logs for debugging
        return None

# --- Image Preprocessing ---
def preprocess_image(image_pil, target_size, expected_channels):
    """Preprocesses the uploaded PIL image for the TensorFlow model."""
    try:
        img = image_pil.resize(target_size)
        if expected_channels == 1 and img.mode != 'L':
            img = img.convert('L')
        elif expected_channels == 3 and img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img)
        # --- Ensure Normalization Matches Colab ---
        # Your Colab uses rescale=1./255 in the generator
        img_array = img_array.astype('float32') / 255.0
        # --- End Normalization ---

        if len(img_array.shape) == 2: # Handle grayscale if needed after conversion
             img_array = np.expand_dims(img_array, axis=-1)

        # Add batch dimension: (h, w, c) -> (1, h, w, c)
        img_array = np.expand_dims(img_array, axis=0)

        st.write(f"Preprocessed image shape: {img_array.shape}")
        return img_array
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

# --- Report Generation ---
def generate_report(predictions, class_names):
    """Generates a text report from model predictions."""
    report_text = "Analysis Report\n"
    report_text += "-----------------\n\n"

    if predictions is None or len(predictions) == 0:
        return report_text + "Could not generate predictions."

    pred_values = predictions[0] # Shape is (1, num_classes), so take the first element

    # --- Check for Class Name Mismatch ---
    if len(pred_values) != len(class_names):
        st.error(f"CRITICAL ERROR: Prediction output size ({len(pred_values)}) does not match number of class names ({len(class_names)}). "
                 f"Please update the CLASS_NAMES list in the Streamlit code (app.py) to match the model's output classes.")
        return report_text + "Error: Configuration mismatch (CLASS_NAMES)."
    # --- End Check ---

    predicted_class_index = np.argmax(pred_values)
    confidence = pred_values[predicted_class_index] * 100
    predicted_class_name = class_names[predicted_class_index]

    report_text += f"Predicted Condition: {predicted_class_name}\n"
    report_text += f"Confidence Score: {confidence:.2f}%\n\n"
    report_text += "Detailed Probabilities:\n"
    for i, class_name in enumerate(class_names):
        report_text += f"- {class_name}: {pred_values[i]*100:.2f}%\n"

    report_text += "\nDisclaimer: This is an automated analysis for informational purposes only. Consult a qualified medical professional for diagnosis."
    return report_text

# --- PDF Generation (Corrected) ---
def create_pdf(text_content):
    """Creates a PDF file in memory from text content."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # Encode carefully to handle potential special characters for PDF generation
        # latin-1 is simple but limited. UTF-8 is broader but needs compatible fonts in FPDF.
        # Using encode/decode('latin-1', 'replace') is a pragmatic way to avoid errors
        # with characters FPDF's default fonts can't handle.
        encoded_text = text_content.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, text=encoded_text)

        # --- FIX: Convert bytearray to bytes ---
        pdf_output_bytearray = pdf.output(dest='b')
        pdf_output_bytes = bytes(pdf_output_bytearray) # Convert to bytes
        # --- End FIX ---

        return pdf_output_bytes # Return bytes object
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        st.exception(e) # Log full error
        return None

# --- Streamlit App ---
st.set_page_config(page_title="CT Scan Analysis", layout="wide")
st.title("🩺 CT Scan Analysis Assistant")
st.write("Upload a CT Scan image (JPG, PNG, BMP) for automated analysis.")

# Load the model
model = load_my_model(MODEL_PATH)

uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None and model is not None:
    try:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded CT Scan', use_column_width=True)

        # Process the image when the button is clicked
        if st.button("Analyze Scan"):
            with col2: # Show processing status and results in the second column
                with st.spinner('Preprocessing image and running analysis...'):
                    preprocessed_img = preprocess_image(image, IMAGE_SIZE, EXPECTED_CHANNELS)

                    if preprocessed_img is not None:
                        try:
                            predictions = model.predict(preprocessed_img)
                            st.success('Analysis Complete!')

                            report = generate_report(predictions, CLASS_NAMES)

                            st.subheader("Analysis Report")
                            st.text_area("Report Content", report, height=300)

                            # Generate PDF only if report generation was successful (no CLASS_NAMES error)
                            if "Error: Configuration mismatch" not in report:
                                pdf_bytes = create_pdf(report)
                                if pdf_bytes:
                                    st.download_button(
                                        label="Download Report as PDF",
                                        data=pdf_bytes, # Now passing 'bytes' object
                                        file_name=f"ct_scan_report_{uploaded_file.name.split('.')[0]}.pdf",
                                        mime="application/pdf"
                                    )
                                else:
                                     st.error("Failed to generate PDF for download.") # Error from create_pdf

                        except Exception as e:
                            st.error(f"Error during prediction or report generation: {e}")
                            st.exception(e)
                    else:
                        st.error("Image preprocessing failed. Cannot analyze.")

    except Exception as e:
        st.error(f"Error opening or processing uploaded image: {e}")
        st.exception(e)

elif uploaded_file is not None and model is None:
    st.warning("Model is not loaded. Cannot analyze. Please check the `MODEL_PATH` and ensure the model file exists and is compatible.")

else:
    st.info("Please upload an image file to begin analysis.")

st.markdown("---")
st.markdown("Developed with Streamlit | Model: MobileNetV2 (CT Scan Analysis)")
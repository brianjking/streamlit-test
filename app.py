import streamlit as st
import torch
from PIL import Image
import pandas as pd
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import tensorflow as tf
import tensorflow_hub as hub
import io
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import logging
import os
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and preprocessors for Image-Text Matching (LAVIS)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model_itm, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)

# Load tokenizer and model for Image Captioning (TextCaps)
git_processor_large_textcaps = AutoProcessor.from_pretrained("microsoft/git-large-r-textcaps")
git_model_large_textcaps = AutoModelForCausalLM.from_pretrained("microsoft/git-large-r-textcaps")

# Load Universal Sentence Encoder model for textual similarity calculation
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Define a function to compute textual similarity between caption and statement
def compute_textual_similarity(caption, statement):
    # Convert caption and statement into sentence embeddings
    caption_embedding = embed([caption])[0].numpy()
    statement_embedding = embed([statement])[0].numpy()

    # Calculate cosine similarity between sentence embeddings
    similarity_score = cosine_similarity([caption_embedding], [statement_embedding])[0][0]
    return similarity_score

# Read statements from the external file 'statements.txt'
with open('statements.txt', 'r') as file:
    statements = file.read().splitlines()

# Function to compute ITM scores for the image-statement pair
def compute_itm_score(image, statement):
    logging.info('Starting compute_itm_score')
    pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
    img = vis_processors["eval"](pil_image.convert("RGB")).unsqueeze(0).to(device)
    # Pass the statement text directly to model_itm
    itm_output = model_itm({"image": img, "text_input": statement}, match_head="itm")
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    score = itm_scores[:, 1].item()
    logging.info('Finished compute_itm_score')
    return score

def generate_caption(processor, model, image):
    logging.info('Starting generate_caption')
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logging.info('Finished generate_caption')
    return generated_caption

def save_dataframe_to_csv(df):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    # Save the CSV string to a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as temp_file:
        temp_file.write(csv_string)
        temp_file_path = temp_file.name  # Get the file path

    # Return the file path (no need to reopen the file with "rb" mode)
    return temp_file_path

       # Define a function to check if the uploaded file is an image
    def is_image_file(file):
        allowed_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
        file_extension = os.path.splitext(file.name)[1]
        return file_extension.lower() in allowed_extensions

    # Main function to perform image captioning and image-text matching
    def process_images_and_statements(file):
        logging.debug("Entered process_images_and_statements function")
        logging.debug(f"File object: {file}")
        logging.debug(f"File name: {file.name}")
        logging.debug(f"File size: {file.tell()}")

        # Check if the uploaded file is an image
        if not is_image_file(file):
            return "Invalid file type. Please upload an image file (e.g., .jpg, .png, .jpeg)."

        # Extract the filename from the file object
        filename = file.name

        # Load the image data from the file (convert file object to bytes using file.read())
        try:
            logging.debug("Attempting to open image")
            image = Image.open(io.BytesIO(file.read()))
            logging.debug("Image opened successfully")
        except Exception as e:
            logging.exception("Error occurred while opening image")
            return str(e)  # Return error message to the user

        # Generate image caption for the uploaded image using git-large-r-textcaps
        caption = generate_caption(git_processor_large_textcaps, git_model_large_textcaps, image)

        # Define weights for combining textual similarity score and image-statement ITM score (adjust as needed)
        weight_textual_similarity = 0.5
        weight_statement = 0.5

        # Initialize an empty list to store the results
        results_list = []

                # Loop through each predefined statement
        for statement in statements:
            # Compute textual similarity between caption and statement
            textual_similarity_score = (compute_textual_similarity(caption, statement) * 100)  # Multiply by 100

            # Compute ITM score for the image-statement pair
            itm_score_statement = (compute_itm_score(image, statement) * 100)  # Multiply by 100

            # Combine the two scores using a weighted average
            final_score = ((weight_textual_similarity * textual_similarity_score) +
                           (weight_statement * itm_score_statement))

            # Append the result to the results_list, including the image filename
            results_list.append({
                'Image Filename': filename,  # Add the image filename to the output
                'Statement': statement,
                'Generated Caption': caption,
                'Textual Similarity Score': f"{textual_similarity_score:.2f}%",  # Format as percentage with two decimal places
                'ITM Score': f"{itm_score_statement:.2f}%",  # Format as percentage with two decimal places
                'Final Combined Score': f"{final_score:.2f}%"  # Format as percentage with two decimal places
            })

        # Convert the results_list to a DataFrame using pandas.concat
        results_df = pd.concat([pd.DataFrame([result]) for result in results_list], ignore_index=True)

        logging.info('Finished process_images_and_statements')

        # Save results_df to a CSV file
        csv_results = save_dataframe_to_csv(results_df)

        # Return both the DataFrame and the CSV data for the Streamlit interface
        return results_df, csv_results

    # Streamlit interface
    def main():
        logging.debug("Entered main() function.")
        st.set_page_config(page_title="Image Captioning and Image-Text Matching", layout="wide")

        st.title("Image Captioning and Image-Text Matching")

        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "gif", "bmp", "tiff"])

        if file is not None:
            # Process the image and generate results
            results_df, csv_results = process_images_and_statements(file)

            # Display results in a table
            st.write(results_df)

            # Provide an option to download the results as a CSV file
            st.markdown(get_csv_download_link(csv_results, file.name), unsafe_allow_html=True)

    def get_csv_download_link(csv_results, filename):
        """Generate a link to download the CSV results."""
        b64 = base64.b64encode(csv_results.encode()).decode()  # Encode to Base64
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}_results.csv">Download CSV</a>'
        return href

    if __name__ == "__main__":
        main()

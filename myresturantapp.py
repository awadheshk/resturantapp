import os
import streamlit as st #streamlit library
import logging
from google.cloud import logging as cloud_logging
import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from datetime import (
    date,
    timedelta,
)
# configure logging
logging.basicConfig(level=logging.INFO)
# attach a Cloud Logging handler to the root logger
log_client = cloud_logging.Client()
log_client.setup_logging()

PROJECT_ID = os.environ.get("GCP_PROJECT")  # Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)


@st.cache_resource
def load_models():
    text_model_flash = GenerativeModel("gemini-2.0-flash-001")
    return text_model_flash


def get_gemini_flash_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)

st.header("My GenAI Resturant recipe helper app", divider="gray")
text_model_flash = load_models()

st.write("Using Gemini Flash - Text only model")
st.subheader("GenAI Cook of My Resturant")

cuisine = st.selectbox(
    "What cuisine do you wish to prepare?",
    ("Indian", "Chinese", "French", "Italian", "Japanese", "Mexican"),
    index=None,
    placeholder="Select the cuisine to be cooked."
)

dietary_preference = st.selectbox(
    "Is there any dietary preferences of the customer?",
    ("Diabetese", "Glueten free", "Halal", "Keto", "Vegan", "Vegetarian", "None"),
    index=None,
    placeholder="Select desired dietary preference (if any)."
)

allergy = st.text_input(
    "Any specific food allergy of the customer:  \n\n", key="allergy", value="peanuts"
)

ingredient_1 = st.text_input(
    "Enter first available item:  \n\n", key="ingredient_1", value="tuna"
)

ingredient_2 = st.text_input(
    "Enter second available item:  \n\n", key="ingredient_2", value="chicken breast"
)

ingredient_3 = st.text_input(
    "Enter third available item:  \n\n", key="ingredient_3", value="tofu"
)

# Complete Streamlit framework code for the user interface, add the wine preference radio button to the interface.
# https://docs.streamlit.io/library/api-reference/widgets/st.radio

wine = st.radio (
    "Do the cusromer is having any wine preference?\n\n", ["Red", "White", "None"], key="wine", horizontal=True
)

max_output_tokens = 2048

# Modify this prompt with the custom prompt.

prompt = f"""I am a resturant Cook.  I need to create {cuisine} \n
recipes for customers who want {dietary_preference} meals. \n
However, don't include recipes that use ingredients with the customer's {allergy} allergy. \n
I have {ingredient_1}, \n
{ingredient_2}, \n
and {ingredient_3} \n
in my kitchen and other ingredients. \n
The customer's wine preference is {wine} \n
Please provide some for meal recommendations.
For each recommendation include preparation instructions, time to prepare and the recipe title at the beginning of the response.
Then include the wine paring for each recommendation.
At the end of the recommendation provide the calories associated with the meal and the nutritional facts.
"""

config = {
    "temperature": 0.8,   # randomness - 0.0 to 1.0
    "max_output_tokens": 2048, // length of the response
}

generate_t2t = st.button("Generate my recipes.", key="generate_t2t")
if generate_t2t and prompt:
    # st.write(prompt)
    with st.spinner("Generating your recipes using Google GenAI (Gemini) ..."):
        first_tab1, first_tab2 = st.tabs(["Recipes", "Prompt"])
        with first_tab1:
            response = get_gemini_flash_text_response(
                text_model_flash,
                prompt,
                generation_config=config,
            )
            if response:
                st.write("Dear Cook, below are recommended recipes:")
                st.write(response)
                logging.info(response)
        with first_tab2:
            st.text(prompt)
          

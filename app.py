import base64
import os

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq


# --- Function to add background image with overlay ---
def add_bg_with_overlay(image_file, overlay_color="rgba(0,0,0,0.5)"):
    """Adds a background image with a semi-transparent overlay."""
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        /* --- Main App Background with Overlay --- */
        .stApp {{
            position: relative;
            background: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            overflow: hidden;
        }}

        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: {overlay_color};
            z-index: 0;
        }}

        /* --- Make all Streamlit content appear above the overlay --- */
        .stApp > * {{
            position: relative;
            z-index: 1;
        }}

        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {{
            background-color: rgba(20, 25, 50, 0.6);
            color: white !important;
        }}

        [data-testid="stSidebar"] * {{
            font-size: 20px !important;
            color: white !important;
        }}

        /* --- Text and Title Styling --- */
        .stMarkdown, .stTitle, .stHeader, .stText {{
            color: white !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# --- Apply background ---
add_bg_with_overlay("images/img.jpg", overlay_color="rgba(0,0,0,0.5)")


# --- Load Groq API key ---
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# --- Initialize LLM ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# --- Define Prompt Templates ---
prompt_name = PromptTemplate.from_template("Suggest a fancy {cuisine} restaurant name.")
prompt_tagline = PromptTemplate.from_template(
    "Write a catchy tagline for a restaurant called '{restaurant_name}'."
)
prompt_menu = PromptTemplate.from_template(
    "Create a detailed {meal_type} menu for a restaurant called '{restaurant_name}' "
    "that serves {cuisine} cuisine and is {dietary} friendly. "
    "Include sections like Appetizers, Main Course, Desserts, and Beverages "
    "with short, appealing descriptions."
)

# --- Define Chains ---
name_chain = RunnableSequence(prompt_name | llm | StrOutputParser())
tagline_chain = RunnableSequence(prompt_tagline | llm | StrOutputParser())
menu_chain = RunnableSequence(prompt_menu | llm | StrOutputParser())

# --- Streamlit UI ---
st.title("🍝 AI Restaurant Generator")
st.write("Generate a restaurant name, tagline, and menu using Groq’s Llama 3 model.")

# --- Sidebar Inputs ---
cuisine = st.sidebar.selectbox(
    "Select a cuisine:",
    [
        "None", "Italian", "Indian", "Japanese", "Chinese", "Mexican", "French",
        "Mediterranean", "Korean", "American", "Angolan", "Cameroonian",
        "Chadian", "Congolese", "Central African", "Equatorial Guinean",
        "Gabonese", "Santomean", "Arabic"
    ]
)

meal_type = st.sidebar.selectbox(
    "Select meal type:",
    ["None", "Appetizer", "Breakfast", "Lunch", "Dinner"]
)

dietary = st.sidebar.selectbox(
    "Select dietary preference:",
    ["None", "Vegan", "Vegetarian", "Jain", "Gluten-Free", "Keto"]
)

# --- Generate Restaurant ---
if st.button("Generate Restaurant"):
    if cuisine == "None":
        st.warning("Please select a cuisine.")
    else:
        with st.spinner("Cooking up your restaurant idea... 🍳"):
            restaurant_name = name_chain.invoke({"cuisine": cuisine})
            tagline = tagline_chain.invoke({"restaurant_name": restaurant_name})
            menu = menu_chain.invoke({
                "meal_type": meal_type,
                "restaurant_name": restaurant_name,
                "cuisine": cuisine,
                "dietary": dietary if dietary != "None" else ""
            })

        st.subheader("🍽️ Restaurant Name")
        st.markdown(f"### {restaurant_name}")

        st.subheader("💬 Tagline")
        st.markdown(f"**{tagline}**")

        st.subheader("📜 Menu")
        st.markdown(menu)

import os

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

# --- Set your Groq API key ---
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

# --- Sidebar inputs ---
cuisine = st.sidebar.selectbox(
    "Select a cuisine:",
    [
        "None","Italian", "Indian", "Japanese", "Chinese", "Mexican", "French",
        "Mediterranean", "Korean", "American", "Angolan", "Cameroonian",
        "Chadian", "Congolese", "Central African", "Equatorial Guinean",
        "Gabonese", "Santomean", "Arabic"
    ]
)

meal_type = st.sidebar.selectbox(
    "Select meal type:",
    ["None","Appetizer","breakfast", "lunch", "dinner"]
)

dietary = st.sidebar.selectbox(
    "Select dietary preference:",
    ["None", "Vegan", "Vegetarian", "Jain", "Gluten-Free", "Keto"]
)

# --- Generate Restaurant ---
if st.button("Generate Restaurant"):
    if not cuisine:
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
        st.markdown(restaurant_name)

        st.subheader("💬 Tagline")
        st.markdown(tagline)

        st.subheader("📜 Menu")
        st.markdown(menu)

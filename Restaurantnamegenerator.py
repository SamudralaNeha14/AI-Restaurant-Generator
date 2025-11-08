import os

os.environ["GROQ_API_KEY"] = "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"

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.6)
response = llm.invoke("Suggest a fancy Italian restaurant name.")

print(response.content)

from langchain_core.prompts import PromptTemplate

prompt_template_name =PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
)
prompt_template_name.format(cuisine="Indian")

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.6)

prompt = PromptTemplate.from_template(
    "Suggest a fancy {cuisine} restaurant name and a short tagline."
)


chain = RunnableSequence(prompt | llm | StrOutputParser())

response = chain.invoke({"cuisine": "Indian"})
print(response)

"""SIMPLE SEQUENTIAL CHAIN"""

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

prompt_name = PromptTemplate.from_template(
    "Suggest a fancy {cuisine} restaurant name."
)
name_chain = RunnableSequence(prompt_name | llm | StrOutputParser())

prompt_tagline = PromptTemplate.from_template(
    "Write a catchy tagline for a restaurant called '{restaurant_name}'."
)
tagline_chain = RunnableSequence(prompt_tagline | llm | StrOutputParser())

prompt_menu = PromptTemplate.from_template(
    "Create a detailed {meal_type} menu for a restaurant called '{restaurant_name}' "
    "that serves {cuisine} cuisine. Include sections like Appetizers, Main Course, Desserts, and Beverages. "
    "Each item should have a short, appealing description."
)
menu_chain = RunnableSequence(prompt_menu | llm | StrOutputParser())

cuisine = "Indian"
meal_type = "lunch"

restaurant_name = name_chain.invoke({"cuisine": cuisine})
tagline = tagline_chain.invoke({"restaurant_name": restaurant_name})
menu = menu_chain.invoke({
    "meal_type": meal_type,
    "restaurant_name": restaurant_name,
    "cuisine": cuisine
})

print("🍽️ Restaurant Name:", restaurant_name)
print("💬 Tagline:", tagline)
print("\n📜 Menu:\n", menu)


import os
# The main change is importing the Gemini model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Set your Google API key from the .env file
# Ensure the environment variable is set for Google, not OpenAI
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini model
# We've updated the model name to a more current and stable version to fix the 404 error.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)

# System prompt (No changes needed here)
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an inventory management expert. Analyze inventory situations and provide clear reorder recommendations"
)

# User prompt template (No changes needed here)
user_prompt = HumanMessagePromptTemplate.from_template(
    """Should I reorder {product}? If so, how much?
    
Current stock: {current_stock} units
Average demand: {average_demand} units per day
Lead time: {lead_time} days

Provide your recommendation and reasoning."""
)

# Create chat prompt (No changes needed here)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

# Create the chain (No changes needed here)
chain = chat_prompt | llm

# Example usage (No changes needed here)
response = chain.invoke({
    "product": "toothpaste",
    "current_stock": 20,
    "average_demand": 5,
    "lead_time": 4
})

print(response.content)

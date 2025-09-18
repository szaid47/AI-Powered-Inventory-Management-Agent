import os
import gradio as gr
# 1. Import the Gemini model instead of OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv

load_dotenv()  

# 2. Set your Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 3. Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)

# System prompt (No changes needed)
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an inventory management expert. Analyze inventory situations and provide clear reorder recommendations with reasoning."
)

# User prompt template (No changes needed)
user_prompt = HumanMessagePromptTemplate.from_template(
    """Should I reorder {product}? If so, how much?
    
Current stock: {current_stock} units
Average demand: {average_demand} units per day
Lead time: {lead_time} days

Provide your recommendation and reasoning."""
)

# Create chat prompt (No changes needed)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

# Create the chain (No changes needed)
chain = chat_prompt | llm

def inventory_chat(product, current_stock, average_demand, lead_time):
    """Function to handle inventory analysis"""
    try:
        response = chain.invoke({
            "product": product,
            "current_stock": int(current_stock),
            "average_demand": float(average_demand),
            "lead_time": int(lead_time)
        })
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface (No changes needed)
with gr.Blocks(title="Inventory Management Assistant") as demo:
    gr.Markdown("# 📦 Inventory Management Assistant")
    gr.Markdown("Get AI-powered recommendations for inventory reordering decisions.")
    
    with gr.Row():
        with gr.Column():
            product_input = gr.Textbox(
                label="Product Name",
                placeholder="e.g., toothpaste",
                value="toothpaste"
            )
            stock_input = gr.Number(
                label="Current Stock (units)",
                value=20,
                minimum=0
            )
            demand_input = gr.Number(
                label="Average Daily Demand (units/day)",
                value=5.0,
                minimum=0.1
            )
            leadtime_input = gr.Number(
                label="Lead Time (days)",
                value=4,
                minimum=1
            )
            
            analyze_btn = gr.Button("Analyze Inventory", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="AI Recommendation",
                lines=15,
                interactive=False
            )
    
    # Connect the function to the button
    analyze_btn.click(
        fn=inventory_chat,
        inputs=[product_input, stock_input, demand_input, leadtime_input],
        outputs=output
    )
    
    # Add example
    gr.Examples(
        examples=[
            ["toothpaste", 20, 5.0, 4],
            ["shampoo", 15, 3.0, 5],
            ["soap", 50, 8.0, 3],
        ],
        inputs=[product_input, stock_input, demand_input, leadtime_input],
    )

if __name__ == "__main__":
    demo.launch()

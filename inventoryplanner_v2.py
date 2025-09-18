import os
import gradio as gr
# 1. Import Gemini model and modern agent creation tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
from database import init_database, get_db_connection  # Assumes local module

# Initialize database
if not init_database():
    raise RuntimeError("Failed to initialize database")

load_dotenv()

# 2. Set your Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 3. Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1, convert_system_message_to_human=True)

# --- Tool Definitions (No changes needed) ---
@tool
def get_inventory_data(product_name: str) -> str:
    """Get inventory data for a specific product from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT product_name, current_stock, average_demand, lead_time
        FROM inventory
        WHERE product_name LIKE ?
    ''', (f'%{product_name}%',))
    result = cursor.fetchone()
    conn.close()
    if result:
        return f"Product: {result[0]}, Current Stock: {result[1]} units, Average Demand: {result[2]} units/day, Lead Time: {result[3]} days"
    else:
        return f"Product '{product_name}' not found in inventory database"

@tool
def get_all_inventory() -> str:
    """Get all inventory data from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT product_name, current_stock, average_demand, lead_time
        FROM inventory
        ORDER BY product_name
    ''')
    results = cursor.fetchall()
    conn.close()
    if results:
        inventory_list = []
        for row in results:
            days_remaining = row[1] / row[2] if row[2] > 0 else 0
            status = "LOW STOCK" if days_remaining <= row[3] else "OK"
            inventory_list.append(f"{row[0]}: {row[1]} units, {days_remaining:.1f} days remaining ({status})")
        return "Current inventory status:\n" + "\n".join(inventory_list)
    else:
        return "No inventory data found"

# --- Agent Setup ---

# System prompt for the agent (No changes needed)
system_prompt = """You are an inventory management expert with access to a real-time inventory database.

When a user asks about inventory decisions:
1. First, retrieve the current inventory data for the requested product using your tools.
2. Analyze the data to determine if reordering is needed.
3. Provide clear recommendations with reasoning.
4. If a user asks a follow-up question, use the context from the conversation history.

Key principles:
- If current stock will last ≤ lead time days, recommend reordering.
- Calculate days remaining: current_stock ÷ average_demand.
- For general inventory questions, use get_all_inventory to show overall status.

Always base your analysis on the actual database data, not assumptions. Be clear and concise.
"""

# 4. Create an agent prompt that includes chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

tools = [get_inventory_data, get_all_inventory]

# 5. Create the agent using the Gemini-compatible function `create_tool_calling_agent`
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Chat Logic and UI ---

def inventory_chat(message, chat_history):
    """Handle chat messages by invoking the agent with conversation history"""
    # Convert Gradio's `List[List[str, str]]` history to LangChain's Message format
    langchain_history = []
    for user_msg, assistant_msg in chat_history:
        langchain_history.append(HumanMessage(content=user_msg))
        langchain_history.append(AIMessage(content=assistant_msg))

    # Invoke the agent executor with the new message and formatted history
    response = agent_executor.invoke({
        "input": message,
        "chat_history": langchain_history
    })
    return response["output"]

# Gradio interface
with gr.Blocks(title="RAG Inventory Management Assistant") as demo:
    gr.Markdown("# 📦 RAG Inventory Management Assistant (Gemini-Powered)")
    gr.Markdown("This assistant uses an AI agent to connect to a real-time inventory database.")

    chatbot = gr.Chatbot(
        label="Inventory Assistant",
        height=400,
        bubble_full_width=False
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Ask about inventory",
            placeholder="e.g., 'Should I reorder toothpaste?'",
            lines=1,
            container=False,
            scale=4,
        )
        submit_btn = gr.Button("Submit", variant="primary", scale=1)
    
    # Use a ClearButton for better UI to clear both the textbox and chatbot
    clear = gr.ClearButton([msg, chatbot])

    gr.Examples(
        examples=[
            "Should I reorder toothpaste?",
            "Check the inventory status for shampoo",
            "Show me all inventory",
            "How many units of soap are left?",
        ],
        inputs=msg,
        label="Example Questions"
    )

    def respond(message, chat_history):
        """UI-facing function to handle message submission"""
        if not message.strip():
            return chat_history, ""

        # Get the bot's response from our agent-powered function
        bot_message = inventory_chat(message, chat_history)
        
        # Append the new interaction to the history
        chat_history.append([message, bot_message])
        
        # Return the updated history for the chatbot and an empty string for the textbox
        return chat_history, ""

    # Connect UI components to the respond function
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])

if __name__ == "__main__":
    demo.launch(share=True)

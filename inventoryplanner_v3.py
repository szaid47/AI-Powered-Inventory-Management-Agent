import os
import gradio as gr
from dotenv import load_dotenv

# 1. Import Gemini model and modern agent creation tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

from database import init_database, get_db_connection

# --- Custom Guardrails (No changes needed) ---
# This class is model-agnostic and works by checking input strings.
class Guardrails:
    def __init__(self):
        self.dangerous_patterns = [
            # Prompt injection attempts
            "ignore previous instructions", "ignore all previous instructions",
            "forget everything above", "forget the previous instructions",
            "new instructions:", "override instructions", "system:", "assistant:",
            "human:", "ignore system prompt",
            
            # Role switching attempts
            "you are now", "act as", "pretend to be", "roleplay as",
            "assume the role", "switch to",
            
            # System prompt extraction
            "what are your instructions", "show me your prompt",
            "repeat your instructions", "what is your system prompt",
            "display your guidelines", "reveal your instructions",
            
            # Command injection
            "<!-->", "<script>", "javascript:", "eval(", "exec(",
            
            # Jailbreak attempts
            "DAN mode", "developer mode", "jailbreak", "unrestricted mode",
        ]
        
    def evaluate(self, user_input):
        """Evaluate user input for security threats."""
        class GuardrailResponse:
            def __init__(self, is_safe=True, filtered_text=None, violation_reason=None):
                self.is_safe = is_safe
                self.filtered_text = filtered_text
                self.violation_reason = violation_reason
        
        if not user_input or not isinstance(user_input, str):
            return GuardrailResponse(is_safe=False, violation_reason="Invalid input format")
            
        user_input_lower = user_input.lower().strip()
        
        for pattern in self.dangerous_patterns:
            if pattern in user_input_lower:
                return GuardrailResponse(is_safe=False, violation_reason=f"Potential security violation detected.")

        if len(user_input) > 2000:
            return GuardrailResponse(is_safe=False, violation_reason="Input is too long.")
        
        return GuardrailResponse(filtered_text=user_input)

# --- App Setup ---
load_dotenv()
if not init_database():
    raise RuntimeError("Failed to initialize database")

# 2. Set your Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

guardrails = Guardrails()

# 3. Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1, convert_system_message_to_human=True)

# --- Tool Definitions (No changes needed) ---
@tool
def get_inventory_data(product_name: str) -> str:
    """Get inventory data for a specific product from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT product_name, current_stock, average_demand, lead_time FROM inventory WHERE product_name LIKE ?', (f'%{product_name}%',))
    result = cursor.fetchone()
    conn.close()
    if result:
        return f"Product: {result[0]}, Current Stock: {result[1]} units, Average Demand: {result[2]} units/day, Lead Time: {result[3]} days"
    else:
        return f"Product '{product_name}' not found in the inventory database."

@tool
def get_all_inventory() -> str:
    """Get all inventory data from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT product_name, current_stock, average_demand, lead_time FROM inventory ORDER BY product_name')
    results = cursor.fetchall()
    conn.close()
    if results:
        inventory_list = [f"{row[0]}: {row[1]} units, {(row[1] / row[2] if row[2] > 0 else 0):.1f} days remaining ({'LOW STOCK' if (row[1] / row[2] if row[2] > 0 else 0) <= row[3] else 'OK'})" for row in results]
        return "Current inventory status:\n" + "\n".join(inventory_list)
    else:
        return "No inventory data found."

# --- Agent Setup ---
system_prompt = """You are an inventory management expert with access to a real-time inventory database.

When a user asks about inventory decisions:
1. First, retrieve the current inventory data for the requested product using your tools.
2. Analyze the data to determine if reordering is needed.
3. Provide clear recommendations with reasoning.
4. If a user asks a follow-up question, use the context from the conversation history.

Always base your analysis on the actual database data. Be clear and concise.
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
    """Handle chat messages, including guardrail checks and agent invocation."""
    # First, validate the input with our security guardrails
    guardrail_response = guardrails.evaluate(message)
    if not guardrail_response.is_safe:
        return f"⚠️ Input blocked: {guardrail_response.violation_reason}"

    # Convert Gradio history to LangChain Message format
    langchain_history = [
        msg for sublist in chat_history for msg in (HumanMessage(content=sublist[0]), AIMessage(content=sublist[1]))
    ]
    
    try:
        # Invoke the agent with the validated message and history
        response = agent_executor.invoke({
            "input": guardrail_response.filtered_text,
            "chat_history": langchain_history
        })
        return response["output"]
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="Secure RAG Inventory Assistant") as demo:
    gr.Markdown("# 📦 Secure RAG Inventory Assistant (Gemini-Powered)")
    gr.Markdown("*Protected by custom input guardrails*")

    chatbot = gr.Chatbot(label="Inventory Assistant", height=400, bubble_full_width=False)
    
    with gr.Row():
        msg = gr.Textbox(label="Ask about inventory", placeholder="e.g., 'Should I reorder toothpaste?'", container=False, scale=4)
        submit_btn = gr.Button("Submit", variant="primary", scale=1)
    
    clear = gr.ClearButton([msg, chatbot])

    gr.Examples(
        examples=["Should I reorder toothpaste?", "Show me all inventory", "How many units of soap are left?"],
        inputs=msg,
        label="Example Questions"
    )

    def respond(message, chat_history):
        """UI-facing function to handle message submission."""
        if not message.strip():
            return chat_history, ""
        bot_message = inventory_chat(message, chat_history)
        chat_history.append([message, bot_message])
        return chat_history, ""

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])

if __name__ == "__main__":
    demo.launch()

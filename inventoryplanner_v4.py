import os
import hashlib
import time
from typing import Dict, Tuple, Optional, List
import gradio as gr
from dotenv import load_dotenv

# 1. Import Gemini model and modern agent creation tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

from database import init_database, get_db_connection

# --- Custom Guardrails Class (No changes needed) ---
# This class is model-agnostic and works by checking input strings.
class Guardrails:
    def __init__(self):
        self.dangerous_patterns = [
            "ignore previous instructions", "ignore all previous instructions",
            "forget everything above", "forget the previous instructions",
            "new instructions:", "override instructions", "system:", "assistant:",
            "human:", "ignore system prompt", "you are now", "act as", "pretend to be",
            "roleplay as", "assume the role", "switch to", "what are your instructions",
            "show me your prompt", "repeat your instructions", "what is your system prompt",
            "display your guidelines", "reveal your instructions", "<!-->", "<script>",
            "javascript:", "eval(", "exec(", "DAN mode", "developer mode",
            "jailbreak", "unrestricted mode",
        ]
        
    def evaluate(self, user_input: str) -> 'GuardrailResponse':
        class GuardrailResponse:
            def __init__(self, is_safe=True, filtered_text=None, violation_reason=None):
                self.is_safe = is_safe
                self.filtered_text = filtered_text
                self.violation_reason = violation_reason
        
        if not user_input or not isinstance(user_input, str):
            return GuardrailResponse(is_safe=False, violation_reason="Invalid input format.")
            
        user_input_lower = user_input.lower().strip()
        
        for pattern in self.dangerous_patterns:
            if pattern in user_input_lower:
                return GuardrailResponse(is_safe=False, violation_reason="Potential security violation detected.")

        if len(user_input) > 2000:
            return GuardrailResponse(is_safe=False, violation_reason="Input is too long.")
        
        return GuardrailResponse(filtered_text=user_input)

# --- Input Cache Class (No changes needed) ---
# This class is also model-agnostic.
class InputCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _generate_hash(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def get(self, input_text: str) -> Optional[str]:
        input_hash = self._generate_hash(input_text)
        if input_hash in self.cache:
            response, timestamp = self.cache[input_hash]
            if time.time() - timestamp <= self.ttl_seconds:
                self.hits += 1
                return response
            else:
                del self.cache[input_hash]
        self.misses += 1
        return None
    
    def set(self, input_text: str, response: str):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        input_hash = self._generate_hash(input_text)
        self.cache[input_hash] = (response, time.time())
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache), "hits": self.hits, "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%", "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }

# --- Application Setup ---
load_dotenv()
if not init_database():
    raise RuntimeError("Failed to initialize database")

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize components
guardrails = Guardrails()
input_cache = InputCache(max_size=500, ttl_seconds=300)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, convert_system_message_to_human=True)

# --- Tool Definitions (No changes needed) ---
@tool
def get_inventory_data(product_name: str) -> str:
    """Use this tool to get specific data for a single product from the inventory database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT product_name, current_stock, average_demand, lead_time FROM inventory WHERE product_name LIKE ?', (f'%{product_name}%',))
    result = cursor.fetchone()
    conn.close()
    if result:
        return f"Data for {result[0]}: Current Stock is {result[1]} units, Average Daily Demand is {result[2]}, Lead Time is {result[3]} days."
    return f"Product '{product_name}' not found."

@tool
def get_all_inventory() -> str:
    """Use this tool to get a summary of all products in the inventory database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT product_name, current_stock, average_demand, lead_time FROM inventory ORDER BY product_name')
    results = cursor.fetchall()
    conn.close()
    if results:
        inventory_list = [f"{row[0]}: {row[1]} units" for row in results]
        return "Full inventory summary:\n" + "\n".join(inventory_list)
    return "No inventory data found."

# 2. CRITICAL FIX: A more direct and forceful prompt to ensure the agent uses its tools.
system_prompt = """You are an inventory management expert. Your primary function is to use tools to access a real-time inventory database and provide analysis.

**Workflow:**
1.  When you receive a question, you **MUST** determine which tool to use: `get_inventory_data` for specific items or `get_all_inventory` for general queries.
2.  After executing the tool, you will receive the data.
3.  You **MUST** base your final answer *only* on the data returned by the tool.
4.  Calculate if a reorder is needed. The reorder point is when (current_stock / average_demand) <= lead_time.
5.  Provide a clear, concise recommendation and the reasoning based on the data. Do not ask the user for information you can find with your tools.
"""

# 3. Create a Gemini-compatible agent with chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

tools = [get_inventory_data, get_all_inventory]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Main Chat Logic ---
def inventory_chat(message: str, chat_history: List[List[str]]):
    # Step 1: Validate input with Guardrails
    guardrail_response = guardrails.evaluate(message)
    if not guardrail_response.is_safe:
        return f"⚠️ Input blocked: {guardrail_response.violation_reason}"
    
    safe_input = guardrail_response.filtered_text

    # Step 2: Check cache
    cached_response = input_cache.get(safe_input)
    if cached_response:
        return f"🔄 {cached_response}"

    # Step 3: If not cached, run the agent
    langchain_history = [
        msg for pair in chat_history for msg in (HumanMessage(content=pair[0]), AIMessage(content=pair[1]))
    ]
    
    try:
        response = agent_executor.invoke({
            "input": safe_input,
            "chat_history": langchain_history
        })
        bot_response = response["output"]
        
        # Step 4: Cache the new response
        input_cache.set(safe_input, bot_response)
        
        return bot_response
    except Exception as e:
        return f"Sorry, an error occurred: {str(e)}"

# --- Gradio UI Helper Functions ---
def format_cache_stats():
    """Helper function to format the cache stats dictionary into a markdown string."""
    stats = input_cache.get_stats()
    return f"""**📊 Cache Statistics**
- **Size**: {stats['size']} / {stats['max_size']}
- **Hits**: {stats['hits']}
- **Misses**: {stats['misses']}
- **Hit Rate**: {stats['hit_rate']}
- **TTL**: {stats['ttl_seconds']}s"""

def clear_and_refresh_stats():
    """Clears the cache and returns the updated stats."""
    input_cache.clear()
    return format_cache_stats()

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Secure RAG Inventory Assistant") as demo:
    gr.Markdown("# 📦 Complete Inventory Agent (Gemini-Powered)")
    gr.Markdown("*Security, Caching, and Real-Time Database Connection*")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Inventory Assistant", height=450, bubble_full_width=False)
            msg = gr.Textbox(label="Ask about inventory", placeholder="e.g., 'Should I reorder toothpaste?'", container=False)
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary", scale=2)
                clear_btn = gr.ClearButton(value="Clear Chat")

        with gr.Column(scale=1):
            gr.Examples(
                examples=["Should I reorder toothpaste?", "How much soap is left?", "Show me all inventory"],
                inputs=msg, label="Example Questions"
            )
            with gr.Group():
                # Corrected line: Use the formatting function and `value` parameter.
                cache_stats_display = gr.Markdown(value=format_cache_stats, label="Cache Stats", every=1)
            with gr.Row():
                refresh_stats_btn = gr.Button("🔄 Refresh")
                clear_cache_btn = gr.Button("🗑️ Clear Cache")

    def respond(message, chat_history):
        if not message.strip():
            return chat_history, ""
        bot_message = inventory_chat(message, chat_history)
        chat_history.append([message, bot_message])
        return chat_history, ""

    # Event handlers
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
    clear_btn.add([chatbot, msg])
    # Corrected event handlers to use the formatting functions
    refresh_stats_btn.click(format_cache_stats, outputs=[cache_stats_display])
    clear_cache_btn.click(clear_and_refresh_stats, outputs=[cache_stats_display])

if __name__ == "__main__":
    demo.launch()


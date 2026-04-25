import inspect

from dotenv import load_dotenv
import re
load_dotenv()



import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL ="qwen3:1.7b"

#too/s (langchain tool decorator) to call the tool in the agent loop

@traceable(name="Get Product Price Tool", run_type="tool")
def get_product_price(product:str)->float:
    """look up the price of a product in the catalog and return the price"""
    # Placeholder implementation - replace with actual product price lookup logic
    print(f"Looking up price for product: '{product}'")
    prices = {"laptop": 999.99, "smartphone": 499.99, "headphones": 199.99, 'Keyboard': 89.99}
    return prices.get(product, 0.0)




@traceable(name="Apply Discount Tool", run_type="tool")
def apply_discount(price:str, discount_tier:str)->float:
    """
    apply a discount tier to a price and return the discounted price 
   Available discount tiers: Silver, gold, Platinum 
   """
    discount_percentage = {"Silver": 10, "gold": 20, "Platinum": 30}
    discount = discount_percentage.get(discount_tier, 0)
    discounted_price = round(float(price) * (1 - discount / 100), 2)
    print(f"Applying {discount_percentage[discount_tier]}% discount to price: {price}. Discounted price: {discounted_price}")
    
    return discounted_price


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,  
}

def get_tool_descriptions(tool_dict):
    
    descriptions =[]

    for tool_name, tool_func in tool_dict.items():
        original_function =getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature}-{docstring}")
        parameters = []
    return ".\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())





#_______________________________React Prompt_______________________________

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

#________________________________________________________





@traceable(name= "Ollama Chat", run_type="llm")
def ollama_chat_traced(MODEL, messages, options):
    return ollama.chat(model=MODEL, messages=messages, options=options)



#____________________Agent Loop ____________________

@traceable(name ="Ollama Agent Loop-> Tool Calling")
# Agent Loop
def run_agent(question:str):
    
    print(f"Question: {question}")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")
        full_prompt = prompt + "\n" + scratchpad

        response = ollama_chat_traced(
            MODEL=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation:"], "temperature": 0},
        )

        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        # CHANGE 6: Parse tool calls from raw text with regex — fragile if LLM doesn't follow format.
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("ERROR: Max iterations reached without a final answer")
    return None
        
        




if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools!)")
    print()
    result = run_agent("what is the price of a laptop and apply a gold discount?")
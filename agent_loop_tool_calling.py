from dotenv import load_dotenv
load_dotenv()

# from langchain.chat_models import init_chat_model
# from langchain.tools import tool
# from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
# from langchain_core.prompts import ChatPromptTemplate    
# from langsmith import traceable


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
def apply_discount(price:float, discount_tier:str)->float:
    """apply a discount tier to a price and return the discounted price"""
    """Available discount tiers: Silver, gold, Platinum"""
    discount_percentage = {"Silver": 10, "gold": 20, "Platinum": 30}
    discount = discount_percentage.get(discount_tier, 0)
    discounted_price = round(price * (1 - discount / 100), 2)
    print(f"Applying {discount_percentage[discount_tier]}% discount to price: {price}. Discounted price: {discounted_price}")
    
    return discounted_price


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name= "Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    response = ollama.chat(MODEL, messages=messages, tools=tools_for_llm)
    return response


@traceable(name ="Ollama Agent Loop-> Tool Calling")
# Agent Loop
def run_agent(question:str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    # llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    # llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    # message =[
    #     SystemMessage(content =(
    #        "You are a helpful shopping assistant. "
    #         "You have access to a product catalog tool "
    #         "and a discount tool.\n\n"
    #         "STRICT RULES — you must follow these exactly:\n"
    #         "1. NEVER guess or assume any product price. "
    #         "You MUST call get_product_price first to get the real price.\n"
    #         "2. Only call apply_discount AFTER you have received "
    #         "a price from get_product_price. Pass the exact price "
    #         "returned by get_product_price — do NOT pass a made-up number.\n"
    #         "3. Never calcluate the duscounted price yourself. You MUST call apply_discount tool to get the discounted price.\n"
    #         "4. if the user doesnt specify a discount tier, ask them which tier to use - do not make assumptions.\n"
    #     )),
    #     HumanMessage(content=question),

    # ]

    messages  = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. Never calcluate the duscounted price yourself. You MUST call apply_discount tool to get the discounted price.\n"
                "4. if the user doesnt specify a discount tier, ask them which tier to use - do not make assumptions.\n"
            )
        },
        {
            "role": "user",
            "content": question
        },
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")

        response = ollama_chat_traced(messages)

        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n final answer: {ai_message.content}")
            # If the model did not call any tools, we assume it has generated a final answer
            return ai_message.content
        
        tool_call = tool_calls[0]  # Get the first tool call
        tool_name = tool_call.function.name  # Get the name of the tool to call
        tool_args = tool_call.function.arguments
        # tool_to_use = tools_dict.get(tool_name)
        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool not found: {tool_name}")
        

        observation = tool_to_use(**tool_args)  # Call the tool with the provided arguments
        print(f"[Tool Result]': {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )
    print("ERROR: Max iterations reached without generating a final answer.")
    return None




if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools!)")
    print()
    result = run_agent("what is the price of a laptop and apply a gold discount?")
#!/usr/bin/env python3
"""
Chat app for interacting with a local vLLM model.
Features:
- Streaming chat interface
- Color-coded messages (thinking, tool calls, regular content)
- Tool selection sidebar
- Fake database management
- Emojis for different message types
"""

import streamlit as st
import json
import re
import hashlib
import argparse
import sys
from typing import List, Dict, Any, Optional
from openai import OpenAI
import time
from datetime import datetime
from datasets import load_dataset

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Chat app for interacting with a local vLLM model")
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8082/v1",
                        help="Base URL for vLLM API")
    parser.add_argument("--vllm-model", type=str, default="qwen3-14b-ft-with-thinking",
                        help="Model name to use with vLLM")
    # Use parse_known_args to ignore Streamlit's arguments
    args, _ = parser.parse_known_args()
    return args

# Initialize OpenAI client for vLLM
def create_client():
    vllm_base_url = st.session_state.get('vllm_base_url', 'http://localhost:8082/v1')
    return OpenAI(
        api_key="dummy-key",
        base_url=vllm_base_url,
        timeout=30.0
    )

# Test connection to vLLM
def test_connection():
    """Test connection to vLLM server."""
    vllm_base_url = st.session_state.get('vllm_base_url', 'http://localhost:8082/v1')
    vllm_model = st.session_state.get('vllm_model', 'qwen3-14b-ft-with-thinking')
    try:
        client = create_client()
        # Try to make a simple chat completion request to test connection
        _ = client.chat.completions.create(
            model=vllm_model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
            timeout=10.0
        )
        return True, f"Connection successful! Server is responding."
    except Exception as e:
        error_details = str(e)
        # Provide more helpful error messages
        if "Connection" in error_details or "refused" in error_details.lower():
            return False, f"❌ Cannot connect to {vllm_base_url}\n\nMake sure vLLM server is running on port 8083.\n\nError: {error_details}"
        elif "timeout" in error_details.lower():
            return False, f"⏱️ Connection timeout to {vllm_base_url}\n\nServer may be overloaded or not responding.\n\nError: {error_details}"
        elif "model" in error_details.lower() or "not found" in error_details.lower():
            return False, f"⚠️ Model '{vllm_model}' not found or unavailable.\n\nError: {error_details}"
        else:
            return False, f"Connection failed: {error_details}"

# Load dataset to extract tools and populate database
@st.cache_data
def load_dataset_info():
    """Load tools and sample data from HuggingFace dataset."""
    tools_list = []
    users = {}
    orders = {}
    products = {}
    
    try:
        # Load dataset from HuggingFace
        ds = load_dataset("Salesforce/APIGen-MT-5k")
        
        # Determine which split to use (prefer 'train', fallback to first available)
        split_name = 'train' if 'train' in ds else list(ds.keys())[0]
        dataset = ds[split_name]
        
        # Filter for retail-related entries and process entries to find retail data
        retail_keywords = ['retail', 'order', 'product', 'customer', 'user_id', 'order_id', 'product_id']
        
        # Process entries to find retail data - process more entries to get representative samples
        # Process up to 2000 entries or entire dataset, whichever is smaller
        max_entries = min(2000, len(dataset))
        
        for i, data in enumerate(dataset):
            if i >= max_entries:
                break
            try:
                # Check if this is a retail-related entry
                system_msg = data.get('system', '').lower()
                is_retail = any(keyword in system_msg for keyword in retail_keywords)
                
                # Also check tool names for retail-related tools
                tools_str = data.get('tools', '[]')
                if isinstance(tools_str, str):
                    tools_raw = json.loads(tools_str)
                else:
                    tools_raw = tools_str
                
                retail_tool_names = ['find_user', 'get_order', 'get_product', 'cancel_order', 'modify_order', 
                                     'return_order', 'exchange_order', 'list_product']
                for tool in tools_raw:
                    if isinstance(tool, dict) and 'name' in tool:
                        tool_name = tool.get('name', '').lower()
                        if any(rt in tool_name for rt in retail_tool_names):
                            is_retail = True
                            break
                
                # Skip non-retail entries
                if not is_retail:
                    continue
                
                # Extract tools
                for tool in tools_raw:
                    if isinstance(tool, dict) and 'name' in tool:
                        tool_name = tool.get('name', '')
                        # Convert to OpenAI format
                        tool_dict = {
                            "type": "function",
                            "function": {
                                "name": tool.get("name", ""),
                                "description": tool.get("description", ""),
                                "parameters": tool.get("parameters", {})
                            }
                        }
                        # Avoid duplicates
                        if not any(t['function']['name'] == tool_name for t in tools_list):
                            tools_list.append(tool_dict)
                
                # Extract data from conversations
                for conv in data.get('conversations', []):
                    if conv['from'] == 'observation':
                        obs = conv['value']
                        try:
                            if isinstance(obs, str):
                                obs_data = json.loads(obs)
                            else:
                                obs_data = obs
                            
                            # Handle both single objects and lists
                            obs_list = obs_data if isinstance(obs_data, list) else [obs_data]
                            
                            for obs_item in obs_list:
                                if not isinstance(obs_item, dict):
                                    continue
                                
                                # Extract user data - handle multiple formats
                                user_id = obs_item.get('user_id') or obs_item.get('id')
                                if user_id:
                                    if user_id not in users:
                                        # Try to extract name from various fields
                                        first_name = obs_item.get('first_name') or obs_item.get('firstname') or obs_item.get('name', {}).get('first', '')
                                        last_name = obs_item.get('last_name') or obs_item.get('lastname') or obs_item.get('name', {}).get('last', '')
                                        
                                        # If name not found, try to parse from user_id
                                        if not first_name and not last_name:
                                            name_parts = str(user_id).split('_')
                                            first_name = name_parts[0].title() if name_parts else 'Unknown'
                                            last_name = name_parts[1].title() if len(name_parts) > 1 else 'User'
                                        
                                        email = obs_item.get('email', f'{user_id}@example.com')
                                        
                                        users[user_id] = {
                                            'user_id': user_id,
                                            'email': email,
                                            'first_name': first_name or 'Unknown',
                                            'last_name': last_name or 'User',
                                            'address': obs_item.get('address', {}),
                                            'payment_methods': obs_item.get('payment_methods', [])
                                        }
                                    else:
                                        # Update existing user with any additional information
                                        existing_user = users[user_id]
                                        if obs_item.get('email') and not existing_user.get('email'):
                                            existing_user['email'] = obs_item.get('email')
                                        if obs_item.get('address') and not existing_user.get('address'):
                                            existing_user['address'] = obs_item.get('address')
                                        if obs_item.get('first_name'):
                                            existing_user['first_name'] = obs_item.get('first_name')
                                        if obs_item.get('last_name'):
                                            existing_user['last_name'] = obs_item.get('last_name')
                                
                                # Extract order data - check multiple possible formats
                                order_id = obs_item.get('order_id') or obs_item.get('id')
                                if order_id:
                                    if order_id not in orders:
                                        order_items = obs_item.get('items', []) or obs_item.get('order_items', [])
                                        # Extract products from order items
                                        for item in order_items:
                                            if isinstance(item, dict):
                                                item_product_id = item.get('product_id') or item.get('id')
                                                item_name = item.get('name') or item.get('product_name') or item.get('product_name')
                                                if item_product_id and item_name:
                                                    if item_product_id not in products:
                                                        products[item_product_id] = {
                                                            'product_id': item_product_id,
                                                            'name': item_name,
                                                            'variants': item.get('variants', {}),
                                                            'price': item.get('price'),
                                                            'description': item.get('description', '')
                                                        }
                                        
                                        orders[order_id] = {
                                            'order_id': order_id,
                                            'user_id': obs_item.get('user_id', ''),
                                            'status': obs_item.get('status', 'pending'),
                                            'items': order_items,
                                            'address': obs_item.get('address', {}),
                                            'payment_history': obs_item.get('payment_history', [])
                                        }
                                    else:
                                        # Update existing order with any additional information
                                        existing_order = orders[order_id]
                                        if obs_item.get('status'):
                                            existing_order['status'] = obs_item.get('status')
                                        if obs_item.get('items'):
                                            existing_order['items'] = obs_item.get('items')
                                        if obs_item.get('user_id') and not existing_order.get('user_id'):
                                            existing_order['user_id'] = obs_item.get('user_id')
                                
                                # Extract product data - check multiple possible formats
                                product_id = obs_item.get('product_id') or obs_item.get('id')
                                product_name = obs_item.get('name') or obs_item.get('product_name') or obs_item.get('title')
                                if product_id and product_name:
                                    if product_id not in products:
                                        products[product_id] = {
                                            'product_id': product_id,
                                            'name': product_name,
                                            'variants': obs_item.get('variants', {}),
                                            'price': obs_item.get('price'),
                                            'description': obs_item.get('description', '') or obs_item.get('desc', '')
                                        }
                                    else:
                                        # Update existing product with any additional information
                                        existing_product = products[product_id]
                                        if obs_item.get('price') and not existing_product.get('price'):
                                            existing_product['price'] = obs_item.get('price')
                                        if obs_item.get('description') and not existing_product.get('description'):
                                            existing_product['description'] = obs_item.get('description')
                                        if obs_item.get('variants'):
                                            existing_product['variants'] = obs_item.get('variants')
                                
                                # Also check if observation contains a list of orders/products
                                if 'orders' in obs_item and isinstance(obs_item['orders'], list):
                                    for order in obs_item['orders']:
                                        if isinstance(order, dict):
                                            oid = order.get('order_id') or order.get('id')
                                            if oid:
                                                if oid not in orders:
                                                    order_items = order.get('items', []) or order.get('order_items', [])
                                                    # Extract products from order items
                                                    for item in order_items:
                                                        if isinstance(item, dict):
                                                            item_product_id = item.get('product_id') or item.get('id')
                                                            item_name = item.get('name') or item.get('product_name')
                                                            if item_product_id and item_name:
                                                                if item_product_id not in products:
                                                                    products[item_product_id] = {
                                                                        'product_id': item_product_id,
                                                                        'name': item_name,
                                                                        'variants': item.get('variants', {}),
                                                                        'price': item.get('price'),
                                                                        'description': item.get('description', '')
                                                                    }
                                                    
                                                    orders[oid] = {
                                                        'order_id': oid,
                                                        'user_id': order.get('user_id', ''),
                                                        'status': order.get('status', 'pending'),
                                                        'items': order_items,
                                                        'address': order.get('address', {}),
                                                        'payment_history': order.get('payment_history', [])
                                                    }
                                
                                if 'products' in obs_item and isinstance(obs_item['products'], list):
                                    for product in obs_item['products']:
                                        if isinstance(product, dict):
                                            pid = product.get('product_id') or product.get('id')
                                            pname = product.get('name') or product.get('product_name') or product.get('title')
                                            if pid and pname:
                                                if pid not in products:
                                                    products[pid] = {
                                                        'product_id': pid,
                                                        'name': pname,
                                                        'variants': product.get('variants', {}),
                                                        'price': product.get('price'),
                                                        'description': product.get('description', '') or product.get('desc', '')
                                                    }
                        except Exception:
                            # Skip if JSON parsing fails - we only want real structured data
                            pass
                    
                    # Extract email from human messages
                    if conv['from'] == 'human':
                        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', conv['value'])
                        if email_match:
                            email = email_match.group()
                            user_id = email.split('@')[0].replace('.', '_')
                            if user_id not in users:
                                name_parts = user_id.split('_')
                                first_name = name_parts[0].title() if name_parts else 'Unknown'
                                last_name = name_parts[1].title() if len(name_parts) > 1 else 'User'
                                users[user_id] = {
                                    'user_id': user_id,
                                    'email': email,
                                    'first_name': first_name,
                                    'last_name': last_name,
                                    'address': {},
                                    'payment_methods': []
                                }
                        
            except Exception:
                continue
        
        # Log extraction results
        import sys
        print(f"[Dataset Loader] Extracted {len(users)} users, {len(orders)} orders, {len(products)} products, {len(tools_list)} tools", file=sys.stderr)
    except Exception as e:
        st.error(f"Error loading dataset from HuggingFace: {str(e)}")
        import sys
        print(f"[Dataset Loader] Error: {str(e)}", file=sys.stderr)
    
    return tools_list, users, orders, products

# Generate sample data with proper relationships
def generate_sample_data():
    """Generate at least 10 users, orders, and products with proper relationships."""
    users = {}
    orders = {}
    products = {}
    
    # Sample product data
    sample_products = [
        {'product_id': 'PROD001', 'name': 'Wireless Headphones', 'price': 79.99, 'description': 'Premium noise-cancelling wireless headphones'},
        {'product_id': 'PROD002', 'name': 'Smart Watch', 'price': 249.99, 'description': 'Fitness tracking smartwatch with heart rate monitor'},
        {'product_id': 'PROD003', 'name': 'Laptop Stand', 'price': 34.99, 'description': 'Adjustable aluminum laptop stand'},
        {'product_id': 'PROD004', 'name': 'USB-C Cable', 'price': 12.99, 'description': '6ft USB-C charging cable'},
        {'product_id': 'PROD005', 'name': 'Wireless Mouse', 'price': 29.99, 'description': 'Ergonomic wireless mouse'},
        {'product_id': 'PROD006', 'name': 'Mechanical Keyboard', 'price': 89.99, 'description': 'RGB mechanical gaming keyboard'},
        {'product_id': 'PROD007', 'name': 'Monitor Stand', 'price': 45.99, 'description': 'Dual monitor stand with cable management'},
        {'product_id': 'PROD008', 'name': 'Webcam HD', 'price': 59.99, 'description': '1080p HD webcam with microphone'},
        {'product_id': 'PROD009', 'name': 'Desk Lamp', 'price': 39.99, 'description': 'LED desk lamp with adjustable brightness'},
        {'product_id': 'PROD010', 'name': 'Phone Stand', 'price': 14.99, 'description': 'Adjustable phone stand for desk'},
        {'product_id': 'PROD011', 'name': 'External Hard Drive', 'price': 89.99, 'description': '1TB portable external hard drive'},
        {'product_id': 'PROD012', 'name': 'Bluetooth Speaker', 'price': 49.99, 'description': 'Portable waterproof Bluetooth speaker'},
    ]
    
    # Sample user data
    sample_users = [
        {'user_id': 'john_smith', 'email': 'john.smith@example.com', 'first_name': 'John', 'last_name': 'Smith', 'zip': '10001'},
        {'user_id': 'jane_doe', 'email': 'jane.doe@example.com', 'first_name': 'Jane', 'last_name': 'Doe', 'zip': '90210'},
        {'user_id': 'bob_johnson', 'email': 'bob.johnson@example.com', 'first_name': 'Bob', 'last_name': 'Johnson', 'zip': '60601'},
        {'user_id': 'alice_williams', 'email': 'alice.williams@example.com', 'first_name': 'Alice', 'last_name': 'Williams', 'zip': '33101'},
        {'user_id': 'charlie_brown', 'email': 'charlie.brown@example.com', 'first_name': 'Charlie', 'last_name': 'Brown', 'zip': '02101'},
        {'user_id': 'diana_prince', 'email': 'diana.prince@example.com', 'first_name': 'Diana', 'last_name': 'Prince', 'zip': '98101'},
        {'user_id': 'edward_miller', 'email': 'edward.miller@example.com', 'first_name': 'Edward', 'last_name': 'Miller', 'zip': '75201'},
        {'user_id': 'fiona_davis', 'email': 'fiona.davis@example.com', 'first_name': 'Fiona', 'last_name': 'Davis', 'zip': '30301'},
        {'user_id': 'george_wilson', 'email': 'george.wilson@example.com', 'first_name': 'George', 'last_name': 'Wilson', 'zip': '19101'},
        {'user_id': 'helen_taylor', 'email': 'helen.taylor@example.com', 'first_name': 'Helen', 'last_name': 'Taylor', 'zip': '94101'},
        {'user_id': 'ivan_martinez', 'email': 'ivan.martinez@example.com', 'first_name': 'Ivan', 'last_name': 'Martinez', 'zip': '78701'},
        {'user_id': 'julia_anderson', 'email': 'julia.anderson@example.com', 'first_name': 'Julia', 'last_name': 'Anderson', 'zip': '80201'},
    ]
    
    # Create products
    for prod in sample_products:
        products[prod['product_id']] = {
            'product_id': prod['product_id'],
            'name': prod['name'],
            'price': prod['price'],
            'description': prod['description'],
            'variants': {}
        }
    
    # Create users with addresses
    for idx, user in enumerate(sample_users):
        users[user['user_id']] = {
            'user_id': user['user_id'],
            'email': user['email'],
            'first_name': user['first_name'],
            'last_name': user['last_name'],
            'address': {
                'address1': f"{123 + idx} Main Street",
                'address2': '',
                'city': 'New York',
                'state': 'NY',
                'zip': user['zip'],
                'country': 'USA'
            },
            'payment_methods': []
        }
    
    # Create orders linked to users and products
    import random
    order_statuses = ['pending', 'processed', 'delivered', 'cancelled']
    user_ids = list(users.keys())
    
    for i in range(12):  # Create 12 orders to ensure we have enough
        order_id = f"#W{1000000 + i}"
        user_id = user_ids[i % len(user_ids)]  # Cycle through users
        
        # Each order has 1-3 products
        num_items = random.randint(1, 3)
        order_items = []
        selected_products = random.sample(sample_products, min(num_items, len(sample_products)))
        
        for prod in selected_products:
            order_items.append({
                'product_id': prod['product_id'],
                'name': prod['name'],
                'quantity': random.randint(1, 2),
                'price': prod['price']
            })
        
        orders[order_id] = {
            'order_id': order_id,
            'user_id': user_id,
            'status': order_statuses[i % len(order_statuses)],
            'items': order_items,
            'address': users[user_id]['address'].copy(),
            'payment_history': []
        }
    
    return users, orders, products

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'enabled_tools' not in st.session_state:
        st.session_state.enabled_tools = []
    if 'users_db' not in st.session_state:
        st.session_state.users_db = {}
    if 'orders_db' not in st.session_state:
        st.session_state.orders_db = {}
    if 'products_db' not in st.session_state:
        st.session_state.products_db = {}
    if 'system_message' not in st.session_state:
        st.session_state.system_message = """# Retail agent policy
As a retail agent, you can help users cancel or modify pending orders, return or exchange delivered orders, modify their default user address, or provide information about their own profile, orders, and related products.
- At the beginning of the conversation, you have to authenticate the user identity by locating their user id via email, or via name + zip code. This has to be done even when the user already provides the user id.
- Once the user has been authenticated, you can provide the user with information about order, product, profile information, e.g. help the user look up order id.
- You can only help one user per conversation (but you can handle multiple requests from the same user), and must deny any requests for tasks related to any other user.
- Before taking consequential actions that update the database (cancel, modify, return, exchange), you have to list the action detail and obtain explicit user confirmation (yes) to proceed.
- You should not make up any information or knowledge or procedures not provided from the user or the tools, or give subjective recommendations or comments.
- You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call.
- You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions."""
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

# Helper function for debug printing
def debug_print(*args, **kwargs):
    """Print debug message only if debug mode is enabled."""
    if st.session_state.get('debug_mode', False):
        st.write(*args, **kwargs)

# Fake database functions
def find_user_by_email(email: str) -> Optional[Dict]:
    """Find user by email."""
    for user in st.session_state.users_db.values():
        if user.get('email') == email:
            return user
    return None

def find_user_by_name_zip(first_name: str, last_name: str, zip_code: str) -> Optional[Dict]:
    """Find user by name and zip."""
    for user in st.session_state.users_db.values():
        if (user.get('first_name', '').lower() == first_name.lower() and
            user.get('last_name', '').lower() == last_name.lower() and
            user.get('address', {}).get('zip') == zip_code):
            return user
    return None

def get_order_details(order_id: str) -> Optional[Dict]:
    """Get order details."""
    return st.session_state.orders_db.get(order_id)

def get_user_details(user_id: str) -> Optional[Dict]:
    """Get user details."""
    return st.session_state.users_db.get(user_id)

def get_product_details(product_id: str) -> Optional[Dict]:
    """Get product details."""
    return st.session_state.products_db.get(product_id)

def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a tool call against the fake database."""
    try:
        if tool_name == "find_user_id_by_email":
            email = arguments.get('email', '')
            user = find_user_by_email(email)
            if user:
                return {"user_id": user['user_id']}
            else:
                return {"error": f"User not found with email: {email}"}
        
        elif tool_name == "find_user_id_by_name_zip":
            first_name = arguments.get('first_name', '')
            last_name = arguments.get('last_name', '')
            zip_code = arguments.get('zip', '')
            user = find_user_by_name_zip(first_name, last_name, zip_code)
            if user:
                return {"user_id": user['user_id']}
            else:
                return {"error": f"User not found with name: {first_name} {last_name} and zip: {zip_code}"}
        
        elif tool_name == "get_order_details":
            order_id = arguments.get('order_id', '')
            order = get_order_details(order_id)
            if order:
                return order
            else:
                return {"error": f"Order not found: {order_id}"}
        
        elif tool_name == "get_user_details":
            user_id = arguments.get('user_id', '')
            user = get_user_details(user_id)
            if user:
                return user
            else:
                return {"error": f"User not found: {user_id}"}
        
        elif tool_name == "get_product_details":
            product_id = arguments.get('product_id', '')
            product = get_product_details(product_id)
            if product:
                return product
            else:
                return {"error": f"Product not found: {product_id}"}
        
        elif tool_name == "list_all_product_types":
            products = []
            for product_id, product in st.session_state.products_db.items():
                products.append({
                    "product_id": product_id,
                    "name": product.get('name', 'Unknown')
                })
            return {"products": products}
        
        elif tool_name == "calculate":
            expression = arguments.get('expression', '')
            try:
                result = eval(expression.replace(' ', ''))
                return {"result": result, "expression": expression}
            except:
                return {"error": f"Invalid expression: {expression}"}
        
        else:
            # For other tools, return a mock response
            return {
                "status": "success",
                "message": f"Tool {tool_name} executed with arguments: {arguments}",
                "note": "This is a mock response. Tool execution not fully implemented."
            }
    
    except Exception as e:
        return {"error": f"Error executing tool: {str(e)}"}

# Stream chat response
def stream_chat_response(messages: List[Dict], enabled_tools: List[Dict]) -> Any:
    """Stream chat response from vLLM."""
    client = create_client()
    
    # Prepare messages for API
    api_messages = []
    for msg in messages:
        if msg['role'] == 'user':
            api_messages.append({"role": "user", "content": msg['content']})
        elif msg['role'] == 'assistant':
            api_msg = {"role": "assistant", "content": msg.get('content', '')}
            if 'tool_calls' in msg:
                api_msg['tool_calls'] = msg['tool_calls']
            if 'thinking' in msg:
                api_msg['reasoning_content'] = msg['thinking']
            api_messages.append(api_msg)
        elif msg['role'] == 'tool':
            api_messages.append({
                "role": "tool",
                "content": json.dumps(msg['content']),
                "tool_call_id": msg.get('tool_call_id', '')
            })
    
    # Prepare API parameters
    vllm_model = st.session_state.get('vllm_model', 'qwen3-14b-ft-with-thinking')
    params = {
        "model": vllm_model,
        "messages": api_messages,
        "temperature": 0.7,
        "max_tokens": 2000,
        "stream": True
    }
    
    if enabled_tools:
        params["tools"] = enabled_tools
        params["reasoning_effort"] = "high"
    
    try:
        vllm_base_url = st.session_state.get('vllm_base_url', 'http://localhost:8082/v1')
        vllm_model = st.session_state.get('vllm_model', 'qwen3-14b-ft-with-thinking')
        debug_print(f"🔍 DEBUG: Calling vLLM at {vllm_base_url} with model {vllm_model}")
        debug_print(f"🔍 DEBUG: API params: {json.dumps({k: v for k, v in params.items() if k != 'messages'}, indent=2)}")
        stream = client.chat.completions.create(**params)
        return stream
    except Exception as e:
        vllm_base_url = st.session_state.get('vllm_base_url', 'http://localhost:8082/v1')
        vllm_model = st.session_state.get('vllm_model', 'qwen3-14b-ft-with-thinking')
        error_msg = f"Error calling vLLM: {str(e)}\n\nBase URL: {vllm_base_url}\nModel: {vllm_model}"
        st.error(error_msg)
        debug_print(f"🔍 DEBUG ERROR: {error_msg}")
        debug_print(f"🔍 DEBUG ERROR: Exception type: {type(e).__name__}")
        import traceback
        debug_print(f"🔍 DEBUG ERROR: Traceback:\n{traceback.format_exc()}")
        return None

# Display message with appropriate styling
def display_message(msg: Dict):
    """Display a message with appropriate styling and emoji."""
    role = msg.get('role', 'user')
    content = msg.get('content', '')
    thinking = msg.get('thinking', '')
    tool_calls = msg.get('tool_calls', [])
    
    debug_print(f"🔍 DEBUG display_message: role={role}, has_thinking={bool(thinking)}, has_content={bool(content)}, has_tool_calls={bool(tool_calls)}")
    
    if role == 'user':
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    
    elif role == 'assistant':
        with st.chat_message("assistant", avatar="🤖"):
            # Display thinking if present (collapsible)
            if thinking:
                debug_print("🔍 DEBUG: Displaying thinking in expander")
                with st.expander("💭 Thinking", expanded=True):
                    # Escape HTML but preserve newlines
                    thinking_escaped = thinking.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                    st.markdown(
                        f'<div style="color: #000000; padding: 12px; background-color: #e3f2fd; border-radius: 5px; border-left: 4px solid #2196f3; white-space: pre-wrap;">{thinking_escaped}</div>',
                        unsafe_allow_html=True
                    )
            
            # Display tool calls if present
            if tool_calls:
                debug_print(f"🔍 DEBUG: Displaying {len(tool_calls)} tool calls")
                for tc in tool_calls:
                    tool_name = tc.get('function', {}).get('name', '') if isinstance(tc.get('function'), dict) else tc.get('name', '')
                    tool_args = tc.get('function', {}).get('arguments', {}) if isinstance(tc.get('function'), dict) else tc.get('arguments', {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except:
                            tool_args = {}
                    
                    st.markdown(f'<div style="color: #e67e22; padding: 10px; background-color: #fff3e0; border-radius: 5px; margin-bottom: 10px;">🔧 <strong>Tool Call:</strong> {tool_name}<br><pre style="font-size: 0.9em;">{json.dumps(tool_args, indent=2)}</pre></div>', unsafe_allow_html=True)
            
            # Display regular content - handle thinking tags
            if content:
                thinking_tag_patterns = ['<think>', '<think>', '<think>', '<think>', '<think>']
                has_thinking_tags = any(tag in content for tag in thinking_tag_patterns)
                debug_print(f"🔍 DEBUG: Displaying content, length={len(content)}, has_thinking_tags={has_thinking_tags}")
                if has_thinking_tags:
                    debug_print(f"🔍 DEBUG: Content preview (first 200 chars): {content[:200]}")
                # Check if content contains thinking tags
                if has_thinking_tags:
                    # Split content into parts and style differently
                    import re
                    # Pattern to match various thinking tags (including redacted_reasoning)
                    pattern = r'(<think>.*?</think>|<think>.*?</think>|<think>.*?</think>|<think>.*?</think>|<think>.*?</think>)'
                    parts = re.split(pattern, content, flags=re.DOTALL)
                    
                    styled_parts = []
                    for i, part in enumerate(parts):
                        if any(part.startswith(tag) for tag in thinking_tag_patterns):
                            # This is thinking content - light blue background
                            debug_print(f"🔍 DEBUG: Part {i} is thinking content, length={len(part)}, starts with: {part[:50]}")
                            # Don't escape - keep HTML tags for proper rendering
                            styled_parts.append(f'<div style="color: #000000; padding: 12px; background-color: #e3f2fd; border-radius: 5px; border-left: 4px solid #2196f3; margin-bottom: 10px; white-space: pre-wrap;">{part}</div>')
                        elif part.strip():
                            # Regular content - light green background
                            debug_print(f"🔍 DEBUG: Part {i} is regular content, length={len(part)}")
                            styled_parts.append(f'<div style="color: #000000; padding: 12px; background-color: #e8f5e9; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;">{part}</div>')
                    
                    st.markdown(''.join(styled_parts), unsafe_allow_html=True)
                else:
                    # No thinking tags, just regular content
                    st.markdown(f'<div style="color: #000000; padding: 12px; background-color: #e8f5e9; border-radius: 5px; white-space: pre-wrap;">{content}</div>', unsafe_allow_html=True)

def main():
    """Main application."""
    # Parse command-line arguments
    args = parse_args()
    
    st.set_page_config(
        page_title="Retail Customer Service Chat App",
        page_icon="💬",
        layout="wide"
    )
    
    init_session_state()
    
    # Store configuration in session state (only on first run)
    if 'vllm_base_url' not in st.session_state:
        st.session_state.vllm_base_url = args.vllm_base_url
        st.session_state.vllm_model = args.vllm_model
        # Print configuration on startup (for debugging)
        import sys
        print(f"[Chat App] Starting with configuration:", file=sys.stderr)
        print(f"[Chat App]   VLLM_BASE_URL: {st.session_state.vllm_base_url}", file=sys.stderr)
        print(f"[Chat App]   VLLM_MODEL: {st.session_state.vllm_model}", file=sys.stderr)
    
    # Load dataset info
    tools_list, users_data, orders_data, products_data = load_dataset_info()
    
    # Initialize databases if empty
    if not st.session_state.users_db:
        st.session_state.users_db = users_data
    if not st.session_state.orders_db:
        st.session_state.orders_db = orders_data
    if not st.session_state.products_db:
        st.session_state.products_db = products_data
    
    # Ensure we have at least 10 users, orders, and products
    # Generate sample data if needed
    sample_users, sample_orders, sample_products = generate_sample_data()
    
    # Merge sample data with existing data, ensuring minimum counts
    if len(st.session_state.users_db) < 10:
        # Add sample users (will overwrite if keys match, but that's fine)
        for user_id, user_data in sample_users.items():
            if user_id not in st.session_state.users_db:
                st.session_state.users_db[user_id] = user_data
    
    if len(st.session_state.products_db) < 10:
        # Add sample products
        for product_id, product_data in sample_products.items():
            if product_id not in st.session_state.products_db:
                st.session_state.products_db[product_id] = product_data
    
    if len(st.session_state.orders_db) < 10:
        # Add sample orders (linked to users and products)
        for order_id, order_data in sample_orders.items():
            if order_id not in st.session_state.orders_db:
                # Ensure the order's user_id exists in users_db
                if order_data['user_id'] not in st.session_state.users_db:
                    # Link to first available user if the original user doesn't exist
                    if st.session_state.users_db:
                        first_user_id = list(st.session_state.users_db.keys())[0]
                        order_data['user_id'] = first_user_id
                        order_data['address'] = st.session_state.users_db[first_user_id]['address'].copy()
                
                # Filter order items to only include products that exist in products_db
                valid_items = []
                for item in order_data.get('items', []):
                    product_id = item.get('product_id')
                    if product_id and product_id in st.session_state.products_db:
                        valid_items.append(item)
                
                # Only add order if it has valid items and a valid user
                if valid_items and order_data['user_id'] in st.session_state.users_db:
                    order_data['items'] = valid_items
                    st.session_state.orders_db[order_id] = order_data
    
    # Sidebar for tools and database management
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.get('debug_mode', False), help="Enable debug printing to see detailed information about message processing")
        
        st.divider()
        
        # Tool selection
        st.subheader("🔧 Tools")
        
        # Tool information section with dropdown
        st.markdown("**📖 Tool Information**")
        tool_names = [t['function']['name'] for t in tools_list]
        selected_tool_name = st.selectbox(
            "Select a tool to view details",
            [""] + tool_names,
            key="tool_info_dropdown",
            help="Select a tool to see its description and parameters"
        )
        
        if selected_tool_name:
            # Find the selected tool
            selected_tool = next((t for t in tools_list if t['function']['name'] == selected_tool_name), None)
            if selected_tool:
                tool_func = selected_tool['function']
                tool_desc = tool_func.get('description', 'No description available')
                tool_params = tool_func.get('parameters', {})
                
                # Display description
                st.markdown(f"**Description:**")
                st.info(tool_desc)
                
                # Display parameters
                if tool_params and 'properties' in tool_params:
                    st.markdown("**Parameters:**")
                    required_params = tool_params.get('required', [])
                    
                    for param_name, param_details in tool_params['properties'].items():
                        param_type = param_details.get('type', 'string')
                        param_desc = param_details.get('description', 'No description')
                        is_required = param_name in required_params
                        
                        # Create a styled parameter display
                        req_badge = "🔴 Required" if is_required else "🟢 Optional"
                        st.markdown(f"""
                        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 8px; border-left: 4px solid {'#e74c3c' if is_required else '#2ecc71'};">
                            <strong>{param_name}</strong> <span style="color: #7f8c8d;">({param_type})</span> {req_badge}<br>
                            <span style="color: #555; font-size: 0.9em;">{param_desc}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("This tool has no parameters.")
        
        st.divider()
        
        # Tool selection checkboxes
        st.markdown("**Enable/Disable Tools**")
        
        # Initialize enabled tools in session state if not set
        if 'tool_checkboxes' not in st.session_state:
            # Default: all tools enabled
            st.session_state.tool_checkboxes = {name: True for name in tool_names}
        else:
            # Ensure all current tools are in the dict
            for name in tool_names:
                if name not in st.session_state.tool_checkboxes:
                    st.session_state.tool_checkboxes[name] = False
        
        # Create checkboxes for each tool with hover info
        selected_tools = []
        for tool in tools_list:
            tool_name = tool['function']['name']
            tool_desc = tool['function'].get('description', 'No description available')
            tool_params = tool['function'].get('parameters', {})
            
            # Format parameters info
            params_info = []
            if tool_params and 'properties' in tool_params:
                for param_name, param_details in tool_params['properties'].items():
                    param_type = param_details.get('type', 'string')
                    param_desc = param_details.get('description', '').replace('"', '&quot;').replace("'", "&#39;")
                    required = param_name in tool_params.get('required', [])
                    req_marker = " (required)" if required else " (optional)"
                    params_info.append(f"• <b>{param_name}</b> ({param_type}){req_marker}: {param_desc}")
            
            params_text = "<br>".join(params_info) if params_info else "No parameters"
            
            # Escape HTML in description
            tool_desc_escaped = tool_desc.replace('"', '&quot;').replace("'", "&#39;").replace('<', '&lt;').replace('>', '&gt;')
            tool_name_escaped = tool_name.replace('"', '&quot;').replace("'", "&#39;")
            
            # Create unique tooltip ID
            tooltip_id = f"tooltip_{hashlib.md5(tool_name.encode()).hexdigest()[:8]}"
            
            # Create tooltip HTML
            tooltip_html = f"""
            <div style="position: relative; display: inline-block;">
                <span style="cursor: help; color: #1f77b4; font-size: 0.9em; margin-left: 5px;" 
                      onmouseover="document.getElementById('{tooltip_id}').style.display='block'"
                      onmouseout="document.getElementById('{tooltip_id}').style.display='none'">ℹ️</span>
                <div id="{tooltip_id}" style="display: none; position: absolute; z-index: 1000; background-color: #fff; 
                     border: 1px solid #ccc; border-radius: 5px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                     width: 350px; max-width: 90vw; left: 25px; top: -10px; font-size: 0.85em; line-height: 1.4;">
                    <div style="font-weight: bold; margin-bottom: 5px; color: #333;">{tool_name_escaped}</div>
                    <div style="color: #666; margin-bottom: 8px;">{tool_desc_escaped}</div>
                    <div style="color: #333; font-weight: bold; margin-top: 8px; margin-bottom: 5px;">Parameters:</div>
                    <div style="color: #555;">{params_text}</div>
                </div>
            </div>
            """
            
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                checked = st.checkbox(
                    tool_name,
                    value=st.session_state.tool_checkboxes.get(tool_name, False),
                    key=f"tool_{tool_name}"
                )
            with col2:
                st.markdown(tooltip_html, unsafe_allow_html=True)
            
            st.session_state.tool_checkboxes[tool_name] = checked
            if checked:
                selected_tools.append(tool_name)
        
        st.session_state.enabled_tools = [t for t in tools_list if t['function']['name'] in selected_tools]
        
        st.divider()
        
        # Database management
        st.subheader("📊 Database Management")
        
        # Users
        with st.expander("👥 Users", expanded=False):
            st.write(f"**Total Users:** {len(st.session_state.users_db)}")
            if st.button("Add User", key="add_user"):
                st.session_state.show_add_user = True
            
            if st.session_state.get('show_add_user', False):
                with st.form("add_user_form"):
                    user_id = st.text_input("User ID")
                    email = st.text_input("Email")
                    first_name = st.text_input("First Name")
                    last_name = st.text_input("Last Name")
                    address1 = st.text_input("Address 1")
                    address2 = st.text_input("Address 2")
                    city = st.text_input("City")
                    state = st.text_input("State")
                    zip_code = st.text_input("Zip Code")
                    submitted = st.form_submit_button("Add")
                    if submitted and user_id:
                        st.session_state.users_db[user_id] = {
                            'user_id': user_id,
                            'email': email or f"{user_id}@example.com",
                            'first_name': first_name or 'Unknown',
                            'last_name': last_name or 'User',
                            'address': {
                                'address1': address1,
                                'address2': address2,
                                'city': city,
                                'state': state,
                                'zip': zip_code,
                                'country': 'USA'
                            },
                            'payment_methods': []
                        }
                        st.session_state.show_add_user = False
                        st.rerun()
            
            # List users
            if st.session_state.users_db:
                user_ids = list(st.session_state.users_db.keys())[:10]
                selected_user = st.selectbox("View User", [""] + user_ids)
                if selected_user:
                    user = st.session_state.users_db[selected_user]
                    st.json(user)
        
        # Orders
        with st.expander("📦 Orders", expanded=False):
            st.write(f"**Total Orders:** {len(st.session_state.orders_db)}")
            if st.button("Add Order", key="add_order"):
                st.session_state.show_add_order = True
            
            if st.session_state.get('show_add_order', False):
                with st.form("add_order_form"):
                    order_id = st.text_input("Order ID (e.g., #W1234567)")
                    user_id = st.text_input("User ID")
                    status = st.selectbox("Status", ["pending", "processed", "delivered", "cancelled"])
                    submitted = st.form_submit_button("Add")
                    if submitted and order_id:
                        st.session_state.orders_db[order_id] = {
                            'order_id': order_id,
                            'user_id': user_id,
                            'status': status,
                            'items': [],
                            'address': {},
                            'payment_history': []
                        }
                        st.session_state.show_add_order = False
                        st.rerun()
            
            # List orders
            if st.session_state.orders_db:
                order_ids = list(st.session_state.orders_db.keys())[:10]
                selected_order = st.selectbox("View Order", [""] + order_ids)
                if selected_order:
                    order = st.session_state.orders_db[selected_order]
                    st.json(order)
        
        # Products
        with st.expander("🛍️ Products", expanded=False):
            st.write(f"**Total Products:** {len(st.session_state.products_db)}")
            if st.button("Add Product", key="add_product"):
                st.session_state.show_add_product = True
            
            if st.session_state.get('show_add_product', False):
                with st.form("add_product_form"):
                    product_id = st.text_input("Product ID")
                    name = st.text_input("Product Name")
                    submitted = st.form_submit_button("Add")
                    if submitted and product_id:
                        st.session_state.products_db[product_id] = {
                            'product_id': product_id,
                            'name': name,
                            'variants': {}
                        }
                        st.session_state.show_add_product = False
                        st.rerun()
            
            # List products
            if st.session_state.products_db:
                product_ids = list(st.session_state.products_db.keys())[:10]
                selected_product = st.selectbox("View Product", [""] + product_ids)
                if selected_product:
                    product = st.session_state.products_db[selected_product]
                    st.json(product)
        
        st.divider()
        
        # Database statistics
        st.subheader("📊 Database Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Users", len(st.session_state.users_db))
        with col2:
            st.metric("Orders", len(st.session_state.orders_db))
        with col3:
            st.metric("Products", len(st.session_state.products_db))
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.title("💬 Retail Customer Service Chat App")
    
    # Connection status and test
    vllm_base_url = st.session_state.get('vllm_base_url', 'http://localhost:8082/v1')
    vllm_model = st.session_state.get('vllm_model', 'qwen3-14b-ft-with-thinking')
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption(f"**Base URL:** `{vllm_base_url}` | **Model:** `{vllm_model}`")
    with col2:
        if st.button("🔌 Test Connection", use_container_width=True):
            with st.spinner("Testing connection..."):
                success, message = test_connection()
                if success:
                    st.success(message)
                else:
                    st.error(message)
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg)
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        display_message(st.session_state.messages[-1])
        
        # Prepare messages for API
        api_messages = [{"role": "system", "content": st.session_state.system_message}]
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                api_messages.append({"role": "user", "content": msg['content']})
            elif msg['role'] == 'assistant':
                api_msg = {"role": "assistant", "content": msg.get('content', '')}
                if 'tool_calls' in msg:
                    api_msg['tool_calls'] = msg['tool_calls']
                if 'thinking' in msg:
                    api_msg['reasoning_content'] = msg['thinking']
                api_messages.append(api_msg)
            elif msg['role'] == 'tool':
                api_messages.append({
                    "role": "tool",
                    "content": json.dumps(msg['content']),
                    "tool_call_id": msg.get('tool_call_id', '')
                })
        
        # Stream response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            thinking_placeholder = st.empty()
            tool_call_placeholder = st.empty()
            full_response = ""
            thinking_content = ""
            tool_calls = []
            
            stream = stream_chat_response(api_messages, st.session_state.enabled_tools)
            
            if stream:
                debug_print("🔍 DEBUG: Starting to process stream chunks")
                for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        
                        # Extract thinking/reasoning
                        if hasattr(delta, 'reasoning') and delta.reasoning:
                            debug_print(f"🔍 DEBUG: Found delta.reasoning, length={len(delta.reasoning)}")
                            thinking_content += delta.reasoning
                            # Escape HTML but preserve newlines
                            thinking_escaped = thinking_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                            thinking_placeholder.markdown(
                                f'<details open style="margin-bottom: 10px; border: 1px solid #90caf9;"><summary style="cursor: pointer; color: #000000; font-weight: bold; padding: 8px; background-color: #bbdefb; user-select: none;">💭 Thinking</summary><div style="color: #000000; padding: 12px; background-color: #e3f2fd; border-radius: 0 0 5px 5px; border-left: 4px solid #2196f3; white-space: pre-wrap;">{thinking_escaped}</div></details>',
                                unsafe_allow_html=True
                            )
                        elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            debug_print(f"🔍 DEBUG: Found delta.reasoning_content, length={len(delta.reasoning_content)}")
                            thinking_content += delta.reasoning_content
                            # Escape HTML but preserve newlines
                            thinking_escaped = thinking_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                            thinking_placeholder.markdown(
                                f'<details open style="margin-bottom: 10px; border: 1px solid #90caf9;"><summary style="cursor: pointer; color: #000000; font-weight: bold; padding: 8px; background-color: #bbdefb; user-select: none;">💭 Thinking</summary><div style="color: #000000; padding: 12px; background-color: #e3f2fd; border-radius: 0 0 5px 5px; border-left: 4px solid #2196f3; white-space: pre-wrap;">{thinking_escaped}</div></details>',
                                unsafe_allow_html=True
                            )
                        elif hasattr(delta, 'reasoning') or hasattr(delta, 'reasoning_content'):
                            # Only log this once to reduce spam
                            if not hasattr(st.session_state, '_reasoning_attr_logged'):
                                debug_print(f"🔍 DEBUG: Has reasoning attributes but values are empty/falsy - thinking content must be in delta.content")
                                st.session_state._reasoning_attr_logged = True
                        
                        # Extract tool calls
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                while len(tool_calls) <= idx:
                                    tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                
                                if tc_delta.id:
                                    tool_calls[idx]["id"] = tc_delta.id
                                if hasattr(tc_delta, 'function'):
                                    if tc_delta.function.name:
                                        tool_calls[idx]["function"]["name"] = tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments
                                
                                # Display tool call
                                tool_name = tool_calls[idx]["function"]["name"]
                                tool_args_str = tool_calls[idx]["function"]["arguments"]
                                try:
                                    tool_args = json.loads(tool_args_str) if tool_args_str else {}
                                except:
                                    tool_args = {}
                                
                                tool_call_placeholder.markdown(
                                    f'<div style="color: #e67e22; padding: 10px; background-color: #fff3e0; border-radius: 5px; margin-bottom: 10px;">🔧 <strong>Tool Call:</strong> {tool_name}<br><pre style="font-size: 0.9em;">{json.dumps(tool_args, indent=2)}</pre></div>',
                                    unsafe_allow_html=True
                                )
                        
                        # Extract content
                        if hasattr(delta, 'content') and delta.content:
                            full_response += delta.content
                            
                            # Check if content contains thinking/reasoning tags
                            # Check for any tag containing "think" or "reasoning" (case insensitive)
                            import re
                            has_thinking_tags = bool(re.search(r'<[^>]*(?:think|reasoning)[^>]*>', full_response, re.IGNORECASE))
                            
                            # Debug: Show what we found (only once when we first detect tags)
                            if has_thinking_tags and not hasattr(st.session_state, '_thinking_tags_logged'):
                                # Find all thinking/reasoning tags
                                found_tag_matches = re.findall(r'<([^>]*(?:think|reasoning)[^>]*)>', full_response, re.IGNORECASE)
                                # Show sample of content to see actual tag structure
                                sample_start = full_response.find('<')
                                sample = full_response[sample_start:min(sample_start + 200, len(full_response))] if sample_start >= 0 else full_response[:200]
                                debug_print(f"🔍 DEBUG: Found thinking/reasoning tags: {found_tag_matches[:5]}")  # Show first 5 matches
                                debug_print(f"🔍 DEBUG: Content sample (first 300 chars): {full_response[:300]}")
                                st.session_state._thinking_tags_logged = True
                            
                            if has_thinking_tags:
                                # Pattern to match any tag containing "think" or "reasoning" and its closing tag
                                # This will match <think>...</think>, <think>...</think>, etc.
                                pattern = r'(<[^>]*(?:think|reasoning)[^>]*>.*?</[^>]*(?:think|reasoning)[^>]*>)'
                                parts = re.split(pattern, full_response, flags=re.DOTALL | re.IGNORECASE)
                                
                                styled_parts = []
                                for part in parts:
                                    # Check if this part is a thinking/reasoning tag (case insensitive)
                                    if re.match(r'<[^>]*(?:think|reasoning)[^>]*>', part, re.IGNORECASE):
                                        # This is thinking content - light blue background
                                        styled_parts.append(f'<div style="color: #000000; padding: 12px; background-color: #e3f2fd; border-radius: 5px; border-left: 4px solid #2196f3; margin-bottom: 10px; white-space: pre-wrap;">{part}</div>')
                                    elif part.strip():
                                        # Regular content - light green background
                                        styled_parts.append(f'<div style="color: #000000; padding: 12px; background-color: #e8f5e9; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;">{part}</div>')
                                
                                message_placeholder.markdown(''.join(styled_parts), unsafe_allow_html=True)
                            else:
                                # No thinking tags, just regular content
                                message_placeholder.markdown(f'<div style="color: #000000; padding: 12px; background-color: #e8f5e9; border-radius: 5px; white-space: pre-wrap;">{full_response}</div>', unsafe_allow_html=True)
                
                # Save assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": full_response
                }
                if thinking_content:
                    assistant_msg["thinking"] = thinking_content
                if tool_calls:
                    # Convert to OpenAI format
                    formatted_tool_calls = []
                    for tc in tool_calls:
                        if tc["function"]["name"]:
                            formatted_tool_calls.append({
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"]
                                }
                            })
                    if formatted_tool_calls:
                        assistant_msg["tool_calls"] = formatted_tool_calls
                
                st.session_state.messages.append(assistant_msg)
                
                # Execute tool calls if any
                if tool_calls:
                    tool_results_placeholder = st.empty()
                    tool_results = []
                    
                    for tc in tool_calls:
                        if tc["function"]["name"]:
                            try:
                                tool_args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                            except:
                                tool_args = {}
                            
                            tool_result = execute_tool_call(tc["function"]["name"], tool_args)
                            tool_results.append({
                                "tool_call_id": tc.get("id", ""),
                                "result": tool_result
                            })
                            
                            # Add tool result message
                            st.session_state.messages.append({
                                "role": "tool",
                                "content": tool_result,
                                "tool_call_id": tc.get("id", "")
                            })
                    
                    # Display all tool results
                    with tool_results_placeholder.container():
                        for tr in tool_results:
                            with st.chat_message("tool", avatar="🔧"):
                                st.json(tr["result"])
                    
                    # Continue conversation with tool results
                    # Prepare messages for follow-up
                    followup_messages = [{"role": "system", "content": st.session_state.system_message}]
                    for msg in st.session_state.messages:
                        if msg['role'] == 'user':
                            followup_messages.append({"role": "user", "content": msg['content']})
                        elif msg['role'] == 'assistant':
                            api_msg = {"role": "assistant", "content": msg.get('content', '')}
                            if 'tool_calls' in msg:
                                api_msg['tool_calls'] = msg['tool_calls']
                            if 'thinking' in msg:
                                api_msg['reasoning_content'] = msg['thinking']
                            followup_messages.append(api_msg)
                        elif msg['role'] == 'tool':
                            followup_messages.append({
                                "role": "tool",
                                "content": json.dumps(msg['content']),
                                "tool_call_id": msg.get('tool_call_id', '')
                            })
                    
                    # Get follow-up response
                    followup_placeholder = st.empty()
                    followup_response = ""
                    followup_thinking = ""
                    followup_tool_calls = []
                    
                    followup_stream = stream_chat_response(followup_messages, st.session_state.enabled_tools)
                    
                    if followup_stream:
                        with followup_placeholder.container():
                            thinking_ph = st.empty()
                            tool_call_ph = st.empty()
                            content_ph = st.empty()
                            
                            for chunk in followup_stream:
                                if chunk.choices:
                                    delta = chunk.choices[0].delta
                                    
                                    # Extract thinking
                                    if hasattr(delta, 'reasoning') and delta.reasoning:
                                        debug_print(f"🔍 DEBUG FOLLOWUP: Found delta.reasoning, length={len(delta.reasoning)}")
                                        followup_thinking += delta.reasoning
                                        # Escape HTML but preserve newlines
                                        thinking_escaped = followup_thinking.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                                        thinking_ph.markdown(
                                            f'<details open style="margin-bottom: 10px; border: 1px solid #90caf9;"><summary style="cursor: pointer; color: #000000; font-weight: bold; padding: 8px; background-color: #bbdefb; user-select: none;">💭 Thinking</summary><div style="color: #000000; padding: 12px; background-color: #e3f2fd; border-radius: 0 0 5px 5px; border-left: 4px solid #2196f3; white-space: pre-wrap;">{thinking_escaped}</div></details>',
                                            unsafe_allow_html=True
                                        )
                                    elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                                        debug_print(f"🔍 DEBUG FOLLOWUP: Found delta.reasoning_content, length={len(delta.reasoning_content)}")
                                        followup_thinking += delta.reasoning_content
                                        # Escape HTML but preserve newlines
                                        thinking_escaped = followup_thinking.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                                        thinking_ph.markdown(
                                            f'<details open style="margin-bottom: 10px; border: 1px solid #90caf9;"><summary style="cursor: pointer; color: #000000; font-weight: bold; padding: 8px; background-color: #bbdefb; user-select: none;">💭 Thinking</summary><div style="color: #000000; padding: 12px; background-color: #e3f2fd; border-radius: 0 0 5px 5px; border-left: 4px solid #2196f3; white-space: pre-wrap;">{thinking_escaped}</div></details>',
                                            unsafe_allow_html=True
                                        )
                                    elif hasattr(delta, 'reasoning') or hasattr(delta, 'reasoning_content'):
                                        debug_print(f"🔍 DEBUG FOLLOWUP: Has reasoning attributes but values are empty/falsy")
                                    
                                    # Extract tool calls
                                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                        for tc_delta in delta.tool_calls:
                                            idx = tc_delta.index
                                            while len(followup_tool_calls) <= idx:
                                                followup_tool_calls.append({
                                                    "id": "",
                                                    "type": "function",
                                                    "function": {"name": "", "arguments": ""}
                                                })
                                            
                                            if tc_delta.id:
                                                followup_tool_calls[idx]["id"] = tc_delta.id
                                            if hasattr(tc_delta, 'function'):
                                                if tc_delta.function.name:
                                                    followup_tool_calls[idx]["function"]["name"] = tc_delta.function.name
                                                if tc_delta.function.arguments:
                                                    followup_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments
                                            
                                            # Display tool call
                                            tool_name = followup_tool_calls[idx]["function"]["name"]
                                            tool_args_str = followup_tool_calls[idx]["function"]["arguments"]
                                            try:
                                                tool_args = json.loads(tool_args_str) if tool_args_str else {}
                                            except:
                                                tool_args = {}
                                            
                                            tool_call_ph.markdown(
                                                f'<div style="color: #e67e22; padding: 10px; background-color: #fff3e0; border-radius: 5px; margin-bottom: 10px;">🔧 <strong>Tool Call:</strong> {tool_name}<br><pre style="font-size: 0.9em;">{json.dumps(tool_args, indent=2)}</pre></div>',
                                                unsafe_allow_html=True
                                            )
                                    
                                    # Extract content
                                    if hasattr(delta, 'content') and delta.content:
                                        debug_print(f"🔍 DEBUG FOLLOWUP: Found delta.content, chunk_length={len(delta.content)}, total_length={len(followup_response) + len(delta.content)}")
                                        followup_response += delta.content
                                        # Check if content contains thinking tags
                                        thinking_tag_patterns = ['<think>', '<think>', '<think>', '<think>', '<think>']
                                        has_thinking_tags = any(tag in followup_response for tag in thinking_tag_patterns)
                                        if has_thinking_tags:
                                            debug_print("🔍 DEBUG FOLLOWUP: Content contains thinking tags, applying special styling")
                                            import re
                                            # Pattern to match various thinking tags (including redacted_reasoning)
                                            pattern = r'(<think>.*?</think>|<think>.*?</think>|<think>.*?</think>|<think>.*?</think>|<think>.*?</think>)'
                                            parts = re.split(pattern, followup_response, flags=re.DOTALL)
                                            
                                            styled_parts = []
                                            for part in parts:
                                                if any(part.startswith(tag) for tag in thinking_tag_patterns):
                                                    # This is thinking content - light blue background
                                                    # Don't escape - keep HTML tags for proper rendering
                                                    styled_parts.append(f'<div style="color: #000000; padding: 12px; background-color: #e3f2fd; border-radius: 5px; border-left: 4px solid #2196f3; margin-bottom: 10px; white-space: pre-wrap;">{part}</div>')
                                                elif part.strip():
                                                    # Regular content - light green background
                                                    styled_parts.append(f'<div style="color: #000000; padding: 12px; background-color: #e8f5e9; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;">{part}</div>')
                                            
                                            content_ph.markdown(''.join(styled_parts), unsafe_allow_html=True)
                                        else:
                                            # No thinking tags, just regular content
                                            content_ph.markdown(f'<div style="color: #000000; padding: 12px; background-color: #e8f5e9; border-radius: 5px; white-space: pre-wrap;">{followup_response}</div>', unsafe_allow_html=True)
                            
                            # Save follow-up assistant message
                            if followup_response or followup_thinking or followup_tool_calls:
                                followup_msg = {
                                    "role": "assistant",
                                    "content": followup_response
                                }
                                if followup_thinking:
                                    followup_msg["thinking"] = followup_thinking
                                if followup_tool_calls:
                                    formatted_tool_calls = []
                                    for tc in followup_tool_calls:
                                        if tc["function"]["name"]:
                                            formatted_tool_calls.append({
                                                "id": tc.get("id", ""),
                                                "type": "function",
                                                "function": {
                                                    "name": tc["function"]["name"],
                                                    "arguments": tc["function"]["arguments"]
                                                }
                                            })
                                    if formatted_tool_calls:
                                        followup_msg["tool_calls"] = formatted_tool_calls
                                st.session_state.messages.append(followup_msg)
            else:
                message_placeholder.error("Failed to get response from vLLM. Please check your connection.")

if __name__ == "__main__":
    main()


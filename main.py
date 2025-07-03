#!/usr/bin/env python3
"""
ZUS Coffee Chatbot - Complete FastAPI Application for Vercel Deployment

This application implements all requirements for the Mindhive Assessment:
- Part 1: Sequential Conversation (state management & memory)
- Part 2: Agentic Planning (intent parsing & action selection)
- Part 3: Tool Calling (calculator integration)
- Part 4: Custom API & RAG Integration (products & outlets endpoints)
- Part 5: Unhappy Flows (error handling & security)

Endpoints:
- POST /chat - Main conversation endpoint
- GET /products - Product search via vector store
- GET /outlets - Store location search via Text2SQL
- GET /calculate - Calculator tool
- DELETE /memory - Clear conversation memory
- GET /health - Health check

Author: ZUS Coffee Chatbot Team
"""

import os
import sys
import sqlite3
import json
import pickle
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import ast
import operator

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ML/AI imports
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configuration
class Config:
    """Application configuration"""

    DATA_DIR = Path("data")
    VECTORSTORE_PATH = DATA_DIR / "vectorstore.faiss"
    METADATA_PATH = DATA_DIR / "vectorstore.pkl"
    DB_PATH = DATA_DIR / "db.sqlite"

    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    GROQ_MODEL = "llama-3.1-8b-instant"

    # Safety limits
    MAX_QUERY_LENGTH = 500
    MAX_RESULTS = 10
    DEFAULT_TOP_K = 5

    # Error handling
    GRACEFUL_DEGRADATION = True
    FALLBACK_RESPONSES = True


# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.warning("GROQ_API_KEY not found. Chat functionality will be limited.")
    groq_client = None
else:
    groq_client = Groq(api_key=groq_api_key)

# Initialize FastAPI app
app = FastAPI(
    title="ZUS Coffee Chatbot API",
    version="1.0.0",
    description="Complete chatbot system for ZUS Coffee with RAG, Text2SQL, and tool integration",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Global state
class GlobalState:
    """Manages global application state"""

    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.vectorstore: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.conversation_memory: Dict[str, List[Dict]] = {}
        self.initialized = False

    def reset_memory(self, user_id: str = "default"):
        """Reset conversation memory for a user"""
        self.conversation_memory[user_id] = []

    def add_to_memory(self, user_id: str, message: str, response: str, action: str):
        """Add conversation turn to memory"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []

        self.conversation_memory[user_id].append(
            {
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "bot_response": response,
                "action_taken": action,
            }
        )

        # Keep only last 6 conversations for memory efficiency
        self.conversation_memory[user_id] = self.conversation_memory[user_id][-6:]


state = GlobalState()

# === PYDANTIC MODELS ===


class UserRequest(BaseModel):
    message: str = Field(
        ..., max_length=Config.MAX_QUERY_LENGTH, description="User message"
    )
    user_id: str = Field(default="default", description="User identifier")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Bot response")
    action_taken: str = Field(..., description="Action performed by the bot")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )
    confidence: float = Field(default=0.0, description="Response confidence score")


class ProductQuery(BaseModel):
    query: str = Field(
        ..., max_length=Config.MAX_QUERY_LENGTH, description="Product search query"
    )
    top_k: Optional[int] = Field(
        default=Config.DEFAULT_TOP_K,
        le=Config.MAX_RESULTS,
        description="Number of results",
    )


class ProductResponse(BaseModel):
    summary: str = Field(..., description="AI-generated summary")
    relevant_products: List[Dict] = Field(..., description="Matching products")
    query: str = Field(..., description="Original query")
    total_found: int = Field(..., description="Total products found")


class OutletQuery(BaseModel):
    query: str = Field(
        ...,
        max_length=Config.MAX_QUERY_LENGTH,
        description="Natural language outlet query",
    )


class OutletResponse(BaseModel):
    summary: str = Field(..., description="Query results summary")
    outlets: List[Dict] = Field(..., description="Matching outlets")
    sql_query: str = Field(..., description="Generated SQL query")
    total_found: int = Field(..., description="Total outlets found")


class CalculatorQuery(BaseModel):
    expression: str = Field(..., max_length=200, description="Mathematical expression")


class CalculatorResponse(BaseModel):
    expression: str = Field(..., description="Original expression")
    result: float = Field(..., description="Calculation result")
    explanation: str = Field(..., description="Step-by-step explanation")


# === UTILITY FUNCTIONS ===


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not text:
        return ""

    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\';\\]', "", text.strip())

    # Limit length
    return sanitized[: Config.MAX_QUERY_LENGTH]


def validate_sql_query(query: str) -> bool:
    """Validate that SQL query is safe (read-only)"""
    dangerous_keywords = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "CREATE",
        "EXEC",
    ]
    query_upper = query.upper()
    return not any(keyword in query_upper for keyword in dangerous_keywords)


def safe_json_loads(text: str, default=None):
    """Safely parse JSON with fallback"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


# === CORE BUSINESS LOGIC ===


class VectorStore:
    """Manages FAISS vector store operations"""

    @staticmethod
    def load_vectorstore() -> Tuple[Optional[faiss.Index], List[Dict]]:
        """Load vectorstore and metadata"""
        try:
            if (
                not Config.VECTORSTORE_PATH.exists()
                or not Config.METADATA_PATH.exists()
            ):
                logger.warning("Vector store files not found")
                return None, []

            # Load FAISS index
            index = faiss.read_index(str(Config.VECTORSTORE_PATH))

            # Load metadata
            with open(Config.METADATA_PATH, "rb") as f:
                metadata = pickle.load(f)

            logger.info(f"Loaded vector store with {len(metadata)} items")
            return index, metadata

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None, []

    @staticmethod
    def search_products(query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for products using vector similarity"""
        try:
            if not state.vectorstore or not state.embedding_model:
                return []

            # Create query embedding
            query_embedding = state.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = state.vectorstore.search(
                query_embedding.astype("float32"), top_k * 2
            )

            # Filter for products and return results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= len(state.metadata):
                    continue

                item = state.metadata[idx]
                if item.get("_type") == "product":
                    results.append((item, float(score)))

                if len(results) >= top_k:
                    break

            return results

        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []


class Text2SQL:
    """Handles natural language to SQL conversion for outlets"""

    @staticmethod
    def get_schema_info() -> str:
        """Get database schema information"""
        return """
        Database Schema:
        Table: outlets
        Columns:
        - id: INTEGER PRIMARY KEY
        - name: TEXT (outlet name)
        - location: TEXT (full address)
        """

    @staticmethod
    def generate_sql(query: str) -> str:
        """Generate SQL query from natural language"""
        try:
            if not groq_client:
                # Fallback SQL generation
                return Text2SQL._fallback_sql_generation(query)

            schema = Text2SQL.get_schema_info()

            prompt = f"""
            Convert this natural language query to SQL for the outlets database.
            
            {schema}
            
            Rules:
            1. Only generate SELECT queries
            2. Use LIKE for text matching
            3. Be case-insensitive with LOWER()
            4. Return only the SQL query, no explanation
            
            Query: {query}
            
            SQL:
            """

            response = groq_client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )

            sql_query = response.choices[0].message.content.strip()

            # Clean up the response
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

            # Validate safety
            if not validate_sql_query(sql_query):
                logger.warning(f"Unsafe SQL query blocked: {sql_query}")
                return "SELECT name, location FROM outlets LIMIT 10"

            return sql_query

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return Text2SQL._fallback_sql_generation(query)

    @staticmethod
    def _fallback_sql_generation(query: str) -> str:
        """Fallback SQL generation without LLM"""
        query_lower = query.lower()

        if "petaling jaya" in query_lower or "pj" in query_lower:
            return "SELECT name, location FROM outlets WHERE LOWER(location) LIKE '%petaling jaya%'"
        elif "kuala lumpur" in query_lower or "kl" in query_lower:
            return "SELECT name, location FROM outlets WHERE LOWER(location) LIKE '%kuala lumpur%'"
        elif "selangor" in query_lower:
            return "SELECT name, location FROM outlets WHERE LOWER(location) LIKE '%%selangor%'"
        else:
            return "SELECT name, location FROM outlets LIMIT 10"

    @staticmethod
    def execute_sql(sql_query: str) -> List[Dict]:
        """Execute SQL query and return results"""
        try:
            if not Config.DB_PATH.exists():
                logger.warning("Database file not found")
                return []

            # Validate query safety
            if not validate_sql_query(sql_query):
                logger.warning("Unsafe SQL query blocked")
                return []

            conn = sqlite3.connect(Config.DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(sql_query)
            rows = cursor.fetchall()

            # Convert to list of dicts
            results = [dict(row) for row in rows]

            conn.close()
            return results

        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            return []


class Calculator:
    """Safe calculator tool for mathematical operations"""

    ALLOWED_OPERATORS = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "**": operator.pow,
        "%": operator.mod,
    }

    @staticmethod
    def parse_expression_with_groq(user_input: str) -> str:
        """Use Groq to parse natural language into mathematical expression"""
        try:
            if not groq_client:
                # Fallback to basic parsing
                return Calculator._fallback_expression_parsing(user_input)

            prompt = f"""
            Extract and convert the mathematical expression from this user input into a clean, evaluable mathematical expression.
            
            Rules:
            1. Only return the mathematical expression, no explanation
            2. Use standard operators: +, -, *, /, **, %, ()
            3. Only include numbers and operators
            4. If no mathematical expression is found, return "INVALID"
            5. Convert words to numbers (e.g., "five" -> "5", "ten" -> "10")
            6. Convert operations (e.g., "plus" -> "+", "times" -> "*", "divided by" -> "/")
            
            Examples:
            Input: "what is 5 plus 3"
            Output: 5 + 3
            
            Input: "calculate two times four"
            Output: 2 * 4
            
            Input: "ten divided by two plus one"
            Output: 10 / 2 + 1
            
            Input: "hello world"
            Output: INVALID
            
            User input: {user_input}
            
            Mathematical expression:
            """

            response = groq_client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )

            parsed_expression = response.choices[0].message.content.strip()

            # Clean up the response
            parsed_expression = parsed_expression.replace("```", "").strip()

            # Validate the parsed expression
            if parsed_expression.upper() == "INVALID" or not parsed_expression:
                return "INVALID"

            # Additional sanitization
            parsed_expression = re.sub(r"[^0-9+\-*/().% ]", "", parsed_expression)

            return parsed_expression if parsed_expression.strip() else "INVALID"

        except Exception as e:
            logger.error(f"Error parsing expression with Groq: {e}")
            return Calculator._fallback_expression_parsing(user_input)

    @staticmethod
    def _fallback_expression_parsing(user_input: str) -> str:
        """Fallback expression parsing without LLM"""
        # Basic word-to-number conversion
        word_to_num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
        }

        # Basic operation conversion
        operation_map = {
            "plus": "+",
            "add": "+",
            "added to": "+",
            "minus": "-",
            "subtract": "-",
            "subtracted from": "-",
            "times": "*",
            "multiply": "*",
            "multiplied by": "*",
            "divided by": "/",
            "divide": "/",
            "power": "**",
            "to the power of": "**",
            "modulo": "%",
            "mod": "%",
        }

        # Convert to lowercase for processing
        processed = user_input.lower()

        # Replace words with numbers
        for word, num in word_to_num.items():
            processed = processed.replace(word, num)

        # Replace operation words
        for word, op in operation_map.items():
            processed = processed.replace(word, op)

        # Extract mathematical expression using regex
        math_pattern = r"([\d+\-*/().% ]+)"
        match = re.search(math_pattern, processed)

        if match:
            expression = match.group(1).strip()
            # Clean up extra spaces
            expression = re.sub(r"\s+", " ", expression)
            return expression

        return "INVALID"

    @staticmethod
    def evaluate(expression: str) -> Dict[str, Any]:
        """Safely evaluate mathematical expression"""
        try:
            # First, try to parse the expression using Groq
            parsed_expression = Calculator.parse_expression_with_groq(expression)

            if parsed_expression == "INVALID":
                raise ValueError("Could not extract a valid mathematical expression")

            # Use the parsed expression
            expression = parsed_expression

            # Final sanitization
            expression = re.sub(r"[^0-9+\-*/().% ]", "", expression)

            if not expression.strip():
                raise ValueError("Empty expression after parsing")

            # Use ast.literal_eval for safety
            try:
                # Parse the expression
                node = ast.parse(expression, mode="eval")
                result = Calculator._eval_node(node.body)

                return {
                    "success": True,
                    "result": result,
                    "explanation": f"{expression} = {result}",
                    "parsed_expression": expression,
                }

            except Exception as e:
                raise ValueError(f"Invalid expression: {e}")

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "explanation": "Could not evaluate expression",
            }

    @staticmethod
    def _eval_node(node):
        """Recursively evaluate AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = Calculator._eval_node(node.left)
            right = Calculator._eval_node(node.right)
            op_type = type(node.op)

            if op_type == ast.Add:
                return left + right
            elif op_type == ast.Sub:
                return left - right
            elif op_type == ast.Mult:
                return left * right
            elif op_type == ast.Div:
                if right == 0:
                    raise ValueError("Division by zero")
                return left / right
            elif op_type == ast.Pow:
                return left**right
            elif op_type == ast.Mod:
                return left % right
            else:
                raise ValueError(f"Unsupported operation: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = Calculator._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return operand
            else:
                raise ValueError(f"Unsupported unary operation: {type(node.op)}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")


class ChatbotPlanner:
    """Implements agentic planning and intent recognition"""

    @staticmethod
    def parse_intent(message: str) -> Dict[str, Any]:
        """Parse user intent and extract parameters using Groq"""
        message_lower = message.lower()

        # If Groq is available, use it for intent classification
        if groq_client:
            try:
                return ChatbotPlanner._parse_intent_with_groq(message)
            except Exception as e:
                logger.error(f"Error using Groq for intent parsing: {e}")
                # Fall back to rule-based parsing
                return ChatbotPlanner._parse_intent_fallback(message_lower)
        else:
            # Use fallback rule-based parsing
            return ChatbotPlanner._parse_intent_fallback(message_lower)

    @staticmethod
    def _parse_intent_with_groq(message: str) -> Dict[str, Any]:
        """Use Groq to classify user intent"""
        prompt = f"""
        Classify the user's intent from the following message. Choose EXACTLY ONE intent from this list:

        AVAILABLE INTENTS:
        1. "calculate" - For mathematical calculations, arithmetic, computing numbers
        2. "search_products" - For finding products, cups, mugs, tumblers, bottles, drinkware, prices
        3. "search_outlets" - For finding store locations, outlets, branches, addresses
        4. "general_chat" - For greetings, general questions, anything else

        RULES:
        - Return ONLY a JSON object with "intent", "confidence", and optionally "expression" or "query"
        - For calculate: include "expression" field with the full user message
        - For search_products/search_outlets: include "query" field with the full user message
        - For general_chat: include "query" field with the full user message
        - Confidence should be between 0.1 and 1.0

        EXAMPLES:
        Input: "what is 5 plus 3"
        Output: {{"intent": "calculate", "expression": "what is 5 plus 3", "confidence": 0.9}}

        Input: "show me coffee mugs"
        Output: {{"intent": "search_products", "query": "show me coffee mugs", "confidence": 0.8}}

        Input: "where is the nearest outlet?"
        Output: {{"intent": "search_outlets", "query": "where is the nearest outlet?", "confidence": 0.8}}

        Input: "hello how are you"
        Output: {{"intent": "general_chat", "query": "hello how are you", "confidence": 0.7}}

        User message: "{message}"

        JSON response:
        """

        response = groq_client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )

        response_text = response.choices[0].message.content.strip()

        # Clean up the response and parse JSON
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        try:
            intent_data = json.loads(response_text)

            # Validate the intent
            valid_intents = [
                "calculate",
                "search_products",
                "search_outlets",
                "general_chat",
            ]
            if intent_data.get("intent") not in valid_intents:
                raise ValueError(f"Invalid intent: {intent_data.get('intent')}")

            # Ensure confidence is reasonable
            confidence = float(intent_data.get("confidence", 0.5))
            confidence = max(0.1, min(1.0, confidence))
            intent_data["confidence"] = confidence

            return intent_data

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error parsing Groq intent response: {e}")
            # Fall back to rule-based parsing
            return ChatbotPlanner._parse_intent_fallback(message.lower())

    @staticmethod
    def _parse_intent_fallback(message_lower: str) -> Dict[str, Any]:
        """Fallback rule-based intent parsing"""
        # Mathematical expressions - enhanced detection
        math_keywords = [
            "+",
            "-",
            "*",
            "/",
            "=",
            "calculate",
            "compute",
            "what is",
            "plus",
            "minus",
            "times",
            "divided",
            "multiply",
            "add",
            "subtract",
        ]

        if any(keyword in message_lower for keyword in math_keywords):
            return {
                "intent": "calculate",
                "expression": message_lower,
                "confidence": 0.8,
            }

        # Location/outlet search (check first, more specific)
        location_keywords = [
            "outlet",
            "store",
            "location",
            "branch",
            "where",
            "address",
            "near",
            "find outlets",
            "store addresses",
        ]
        if any(keyword in message_lower for keyword in location_keywords):
            return {
                "intent": "search_outlets",
                "query": message_lower,
                "confidence": 0.7,
            }

        # Product search
        product_keywords = [
            "product",
            "cup",
            "mug",
            "tumbler",
            "bottle",
            "drinkware",
            "buy",
            "price",
        ]
        if any(keyword in message_lower for keyword in product_keywords):
            return {
                "intent": "search_products",
                "query": message_lower,
                "confidence": 0.7,
            }

        # General conversation
        return {"intent": "general_chat", "query": message_lower, "confidence": 0.5}

    @staticmethod
    def plan_action(intent_result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Plan the next action based on intent"""
        intent = intent_result["intent"]

        if intent == "calculate":
            return {
                "action": "calculate",
                "parameters": {"expression": intent_result["expression"]},
                "explanation": "Performing mathematical calculation",
            }
        elif intent == "search_products":
            return {
                "action": "search_products",
                "parameters": {"query": intent_result["query"]},
                "explanation": "Searching for products",
            }
        elif intent == "search_outlets":
            return {
                "action": "search_outlets",
                "parameters": {"query": intent_result["query"]},
                "explanation": "Searching for store locations",
            }
        else:
            return {
                "action": "general_response",
                "parameters": {"message": intent_result["query"]},
                "explanation": "Providing general response",
            }


# === INITIALIZATION ===


def initialize_system():
    """Initialize the chatbot system"""
    try:
        logger.info("Initializing ZUS Coffee Chatbot...")

        # Load embedding model
        logger.info("Loading embedding model...")
        state.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Load vector store
        logger.info("Loading vector store...")
        state.vectorstore, state.metadata = VectorStore.load_vectorstore()

        if state.vectorstore is None:
            logger.warning("Vector store not loaded. Product search will be limited.")

        # Check database
        if not Config.DB_PATH.exists():
            logger.warning("Database not found. Outlet search will be limited.")

        state.initialized = True
        logger.info("System initialization completed successfully")

    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        logger.error(traceback.format_exc())
        state.initialized = False


# === FASTAPI STARTUP ===


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    initialize_system()


# === API ENDPOINTS ===


@app.get("/")
async def root():
    """Serve the main page"""
    try:
        if Path("static/index.html").exists():
            return FileResponse("static/index.html")
        else:
            return {
                "message": "ZUS Coffee Chatbot API",
                "version": "1.0.0",
                "endpoints": [
                    "/chat",
                    "/products",
                    "/outlets",
                    "/calculate",
                    "/health",
                ],
            }
    except Exception as e:
        logger.error(f"Error serving root: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.post("/chat", response_model=ChatResponse)
async def chat(request: UserRequest):
    """Main conversation endpoint with agentic planning"""
    try:
        # Sanitize input
        message = sanitize_input(request.message)
        if not message:
            raise HTTPException(status_code=400, detail="Empty message")

        user_id = request.user_id

        # Parse intent using agentic planner
        intent_result = ChatbotPlanner.parse_intent(message)
        action_plan = ChatbotPlanner.plan_action(intent_result, user_id)

        # Execute planned action
        if action_plan["action"] == "calculate":
            calc_result = Calculator.evaluate(action_plan["parameters"]["expression"])
            if calc_result["success"]:
                parsed_expr = calc_result.get("parsed_expression", "")
                if (
                    parsed_expr
                    and parsed_expr != action_plan["parameters"]["expression"]
                ):
                    response = f"I understood your question as: {parsed_expr}\nThe result is: {calc_result['result']}"
                else:
                    response = f"The result is: {calc_result['result']}\n{calc_result['explanation']}"
                action_taken = "calculator"
            else:
                response = f"I couldn't calculate that: {calc_result['error']}"
                action_taken = "calculator_error"

        elif action_plan["action"] == "search_products":
            products = VectorStore.search_products(action_plan["parameters"]["query"])
            if products:
                product_list = []
                for product, score in products[:3]:
                    price = product.get("price", ["N/A"])
                    price_str = price[0] if isinstance(price, list) else str(price)
                    product_list.append(f"• {product['name']} - {price_str}")

                response = f"I found these products for you:\n" + "\n".join(
                    product_list
                )
                if len(products) > 3:
                    response += f"\n\nAnd {len(products) - 3} more products."
                action_taken = "product_search"
            else:
                response = "I couldn't find any products matching your request. Could you try rephrasing your query?"
                action_taken = "product_search_empty"

        elif action_plan["action"] == "search_outlets":
            sql_query = Text2SQL.generate_sql(action_plan["parameters"]["query"])
            outlets = Text2SQL.execute_sql(sql_query)
            if outlets:
                outlet_list = []
                for outlet in outlets[:3]:
                    outlet_list.append(f"• {outlet['name']}: {outlet['location']}")

                response = f"I found these ZUS Coffee outlets:\n" + "\n".join(
                    outlet_list
                )
                if len(outlets) > 3:
                    response += f"\n\nAnd {len(outlets) - 3} more locations."
                action_taken = "outlet_search"
            else:
                response = "I couldn't find any outlets matching your query. Could you try a different location?"
                action_taken = "outlet_search_empty"

        else:
            # General conversation
            if groq_client:
                try:
                    # Get conversation context
                    context = ""
                    if user_id in state.conversation_memory:
                        recent_context = state.conversation_memory[user_id][
                            -3:
                        ]  # Last 3 turns
                        context = "\n".join(
                            [
                                f"User: {turn['user_message']}\nBot: {turn['bot_response']}"
                                for turn in recent_context
                            ]
                        )

                    prompt = f"""You are a helpful ZUS Coffee chatbot assistant. 
                    Be friendly, concise, and helpful. Focus on ZUS Coffee products and locations. Zus's operating hours are generally 10am to 10pm. But it may vary by location.
                    
                    Previous conversation:
                    {context}
                    
                    Current question: {message}
                    
                    Response:"""

                    groq_response = groq_client.chat.completions.create(
                        model=Config.GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.7,
                    )

                    response = groq_response.choices[0].message.content.strip()
                    action_taken = "general_chat"

                except Exception as e:
                    logger.error(f"Groq API error: {e}")
                    response = "I'm here to help you with ZUS Coffee products and store locations. What would you like to know?"
                    action_taken = "fallback_response"
            else:
                response = "I'm here to help you with ZUS Coffee products and store locations. What would you like to know?"
                action_taken = "fallback_response"

        # Add to memory
        state.add_to_memory(user_id, message, response, action_taken)

        return ChatResponse(
            response=response,
            action_taken=action_taken,
            context={
                "intent": intent_result["intent"],
                "confidence": intent_result["confidence"],
            },
            confidence=intent_result["confidence"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        logger.error(traceback.format_exc())

        # Graceful degradation
        return ChatResponse(
            response="I'm experiencing some technical difficulties. Please try again or contact support if the issue persists.",
            action_taken="error_fallback",
            context={"error": str(e)},
            confidence=0.0,
        )


@app.get("/products", response_model=ProductResponse)
async def search_products(
    query: str = Query(..., description="Product search query"),
    top_k: int = Query(
        default=Config.DEFAULT_TOP_K,
        le=Config.MAX_RESULTS,
        description="Number of results",
    ),
):
    """Product search endpoint using vector store"""
    try:
        # Sanitize input
        query = sanitize_input(query)
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        # Search products
        results = VectorStore.search_products(query, top_k)

        # Extract products
        products = [product for product, score in results]

        # Generate AI summary if possible
        if groq_client and products:
            try:
                product_names = [p["name"] for p in products[:3]]
                summary_prompt = f"Summarize these ZUS Coffee products for the query '{query}': {', '.join(product_names)}"

                groq_response = groq_client.chat.completions.create(
                    model=Config.GROQ_MODEL,
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=100,
                    temperature=0.3,
                )

                summary = groq_response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary = f"Found {len(products)} products matching '{query}'"
        else:
            summary = (
                f"Found {len(products)} products matching '{query}'"
                if products
                else f"No products found for '{query}'"
            )

        return ProductResponse(
            summary=summary,
            relevant_products=products,
            query=query,
            total_found=len(products),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in products endpoint: {e}")
        raise HTTPException(status_code=500, detail="Product search failed")


@app.get("/outlets", response_model=OutletResponse)
async def search_outlets(
    query: str = Query(..., description="Natural language outlet query")
):
    """Outlet search endpoint using Text2SQL"""
    try:
        # Sanitize input
        query = sanitize_input(query)
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")

        # Generate SQL
        sql_query = Text2SQL.generate_sql(query)

        # Execute SQL
        outlets = Text2SQL.execute_sql(sql_query)

        # Generate summary
        if outlets:
            summary = f"Found {len(outlets)} ZUS Coffee outlets matching your query"
        else:
            summary = "No outlets found matching your query"

        return OutletResponse(
            summary=summary,
            outlets=outlets,
            sql_query=sql_query,
            total_found=len(outlets),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in outlets endpoint: {e}")
        raise HTTPException(status_code=500, detail="Outlet search failed")


@app.get("/calculate", response_model=CalculatorResponse)
async def calculate(
    expression: str = Query(..., description="Mathematical expression to evaluate")
):
    """Calculator tool endpoint"""
    try:
        # Sanitize expression
        expression = sanitize_input(expression)
        if not expression:
            raise HTTPException(status_code=400, detail="Empty expression")

        # Calculate
        result = Calculator.evaluate(expression)

        if result["success"]:
            return CalculatorResponse(
                expression=result.get("parsed_expression", expression),
                result=result["result"],
                explanation=result["explanation"],
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in calculate endpoint: {e}")
        raise HTTPException(status_code=500, detail="Calculation failed")


@app.delete("/memory")
async def clear_memory(user_id: str = Query(default="default", description="User ID")):
    """Clear conversation memory for a user"""
    try:
        state.reset_memory(user_id)
        return {"message": f"Memory cleared for user {user_id}"}
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear memory")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy" if state.initialized else "initializing",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "embedding_model": state.embedding_model is not None,
                "vector_store": state.vectorstore is not None,
                "database": Config.DB_PATH.exists(),
                "groq_client": groq_client is not None,
            },
            "data_counts": {
                "total_items": len(state.metadata),
                "products": len(
                    [item for item in state.metadata if item.get("_type") == "product"]
                ),
                "stores": len(
                    [item for item in state.metadata if item.get("_type") == "store"]
                ),
            },
        }

        # Check if all critical components are working
        if not state.initialized:
            health_status["status"] = "unhealthy"
            return JSONResponse(status_code=503, content=health_status)

        return health_status

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )


# === ERROR HANDLERS ===


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions gracefully"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions gracefully"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat(),
        },
    )


# === STARTUP VERIFICATION ===


@app.middleware("http")
async def startup_check_middleware(request: Request, call_next):
    """Middleware to check if system is initialized"""
    # Skip initialization check for health endpoint
    if request.url.path == "/health":
        response = await call_next(request)
        return response

    # Check if system is initialized for other endpoints
    if not state.initialized and request.url.path not in [
        "/",
        "/docs",
        "/openapi.json",
    ]:
        return JSONResponse(
            status_code=503,
            content={
                "error": "System not initialized",
                "message": "The chatbot is still starting up. Please try again in a moment.",
                "timestamp": datetime.now().isoformat(),
            },
        )

    response = await call_next(request)
    return response


# === MAIN ===

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info"
    )

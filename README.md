# ZUS Coffee Chatbot - Technical Assessment

A production-ready chatbot system for ZUS Coffee built with FastAPI, implementing conversational AI patterns including agentic planning, tool integration, and retrieval-augmented generation (RAG). The system is designed for serverless deployment with a focus on maintainability and production reliability.

## Architecture Overview

This system implements a serverless-first chatbot architecture optimized for production deployment:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI App    │    │   Data Layer    │
│   (index.html)  │◄──►│   (main.py)      │◄──►│   (vectorstore, │
│                 │    │                  │    │    sqlite)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   External APIs  │
                    │   (Groq LLM)     │
                    └──────────────────┘
```

### Design Principles

1. **Serverless-First**: Optimized for Vercel deployment with cold start considerations
2. **Fault Tolerance**: Graceful degradation when external services fail
3. **Clean Architecture**: Separation of concerns with validated data schemas
4. **Security**: Input sanitization and safe code execution
5. **Observability**: Structured logging and health monitoring

## Assessment Requirements Implementation

### Part 1: Sequential Conversation
- **Memory Management**: Persistent conversation history per user (last 6 turns)
- **Context Awareness**: Previous conversation context influences responses
- **State Management**: Global state management with user isolation
- **Multi-turn Support**: Seamless conversation flow across interactions

### Part 2: Agentic Planning
- **Intent Classification**: Groq-powered intent recognition with fallback rules
- **Action Planning**: Strategic action selection based on parsed intent
- **Confidence Scoring**: Probabilistic confidence assessment for responses
- **Four Core Intents**: `calculate`, `search_products`, `search_outlets`, `general_chat`

### Part 3: Tool Calling
- **Calculator Tool**: Safe mathematical expression evaluation using AST parsing
- **Natural Language Processing**: Groq-powered expression parsing from natural language
- **Security**: Sandboxed execution preventing code injection
- **Error Handling**: Graceful failure with informative error messages

### Part 4: Custom API & RAG Integration
- **Vector Search**: FAISS-powered semantic search for products
- **Text2SQL**: Natural language to SQL conversion for outlet queries
- **Embeddings**: Sentence-BERT for semantic understanding
- **Database Integration**: SQLite for structured outlet data

### Part 5: Unhappy Flows
- **Input Validation**: Comprehensive sanitization and validation
- **Error Recovery**: Graceful degradation with fallback responses
- **Security**: SQL injection prevention, input sanitization
- **Monitoring**: Health checks and system status endpoints

## Quick Start

### Prerequisites
- Python 3.12+
- Groq API key

### Local Development

1. **Clone and Setup**
```bash
git clone <repository-url>
cd chatbot-engineer-assessment
pip install -r requirements.txt
```

2. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

3. **Run the Application**
```bash
python main.py
```

4. **Access the Chatbot**
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Production Deployment (Vercel)

1. **Push to GitHub**
```bash
git add .
git commit -m "Deploy ZUS Coffee Chatbot"
git push origin main
```

2. **Deploy via Vercel Dashboard**
- Connect your GitHub repository
- Set environment variable: `GROQ_API_KEY`
- Deploy automatically

3. **Verify Deployment**
```bash
curl https://your-deployment.vercel.app/health
```

## Project Structure

```
chatbot-engineer-assessment/
├── main.py                 # Complete FastAPI application
├── requirements.txt        # Python dependencies
├── vercel.json            # Vercel deployment configuration
├── .env.example           # Environment variables template
├── data/                  # Pre-generated data files
│   ├── vectorstore.faiss  # FAISS vector index
│   ├── vectorstore.pkl    # Product metadata
│   └── db.sqlite         # Outlet database
├── static/               # Frontend assets
│   └── index.html        # Web interface
└── README.md            # This file
```

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Example |
|----------|--------|-------------|---------|
| `/chat` | POST | Main conversation endpoint | `{"message": "Calculate 15% of 200"}` |
| `/products` | GET | Product search via vector similarity | `/products?query=coffee mug&top_k=5` |
| `/outlets` | GET | Outlet search via Text2SQL | `/outlets?query=outlets in KL` |
| `/calculate` | GET | Mathematical expression evaluation | `/calculate?expression=5+3*2` |
| `/memory` | DELETE | Clear user conversation memory | `/memory?user_id=user123` |
| `/health` | GET | System health and status | Health monitoring |

### Example API Usage

**Chat Conversation:**
```bash
curl -X POST "https://your-app.vercel.app/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me insulated coffee mugs", "user_id": "user123"}'
```

**Product Search:**
```bash
curl "https://your-app.vercel.app/products?query=travel tumbler&top_k=3"
```

**Outlet Location:**
```bash
curl "https://your-app.vercel.app/outlets?query=find outlets in Petaling Jaya"
```

## Technical Implementation

### Agentic Planning System
```python
# Intent classification with Groq
intent_result = ChatbotPlanner.parse_intent(user_message)
action_plan = ChatbotPlanner.plan_action(intent_result, user_id)

# Available intents:
# - calculate: Mathematical operations
# - search_products: Product discovery
# - search_outlets: Location queries  
# - general_chat: Conversational responses
```

### Vector Search (RAG)
```python
# Semantic product search using FAISS + Sentence-BERT
query_embedding = embedding_model.encode([query])
scores, indices = vectorstore.search(query_embedding, top_k)
```

### Text2SQL Generation
```python
# Natural language to SQL conversion
sql_query = Text2SQL.generate_sql("find outlets in KL")
# Output: "SELECT name, location FROM outlets WHERE LOWER(location) LIKE '%kuala lumpur%'"
```

### Safe Calculator
```python
# Groq-powered expression parsing + AST evaluation
parsed_expr = Calculator.parse_expression_with_groq("fifteen percent of two hundred")
# Output: "15 / 100 * 200"
result = Calculator.evaluate(parsed_expr)
# Output: 30.0
```

## Security & Production Considerations

### Input Validation & Security
- **SQL Injection Prevention**: Parameterized queries with input sanitization
- **Code Injection Protection**: AST-based mathematical expression parsing
- **XSS Prevention**: Input sanitization for all user-generated content
- **CORS Configuration**: Proper cross-origin resource sharing setup

### Error Handling & Resilience
- **Circuit Breaker Pattern**: Graceful degradation when external services fail
- **Comprehensive Logging**: Structured logging for debugging and monitoring
- **User-Friendly Messages**: Error responses without exposing internals
- **Health Monitoring**: Critical component status tracking
- **Fallback Mechanisms**: Rule-based fallbacks when ML models fail

### Performance Optimization
- **Cold Start Optimization**: Efficient initialization for serverless deployment
- **Memory Management**: Conversation memory capping (6 turns per user)
- **Lazy Loading**: ML models loaded on first use
- **Efficient Data Structures**: FAISS for vector search, optimized SQLite queries

## Data Pipeline & Management

### Data Architecture

**Products Schema**:
```python
{
  "name": str,      # Product name
  "price": float,   # Price in RM
  "url": str        # Product URL
}
```

**Stores Schema**:
```python
{
  "name": str,      # Store name
  "location": str   # Store location
}
```

### Data Processing Pipeline
1. **Web Scraping**: Automated data collection from ZUS Coffee website
2. **Data Validation**: Pydantic models ensure schema compliance
3. **Preprocessing**: Text normalization and metadata extraction
4. **Embedding Generation**: Sentence-BERT vectorization
5. **Index Creation**: FAISS index construction for fast retrieval
6. **Database Storage**: SQLite with optimized queries

## Testing Strategy

### Test Coverage
- **Unit Tests**: Calculator, planner, and core business logic
- **Integration Tests**: API endpoints and data pipeline
- **Security Tests**: Input validation and injection prevention

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_calculator_new.py -v
python -m pytest tests/test_planner_new.py -v
```

## System Health Monitoring

The `/health` endpoint provides system status:
```json
{
  "status": "healthy",
  "components": {
    "embedding_model": true,
    "vector_store": true,
    "database": true,
    "groq_client": true
  },
  "data_counts": {
    "total_items": 150,
    "products": 120,
    "stores": 30
  }
}
```

### Logging Strategy
- **INFO**: System initialization and normal operations
- **WARNING**: Fallback activations and degraded performance
- **ERROR**: Failures and exceptions with full stack traces

## Deployment

### Local Development
1. **Environment Setup**: `pip install -r requirements.txt`
2. **Configuration**: Copy `.env.example` to `.env` and configure API keys
3. **Development Server**: `python main.py`
4. **Testing**: Run test suite with `pytest tests/ -v`

### Production Deployment (Vercel)
1. **Push to GitHub**: Deploy repository to GitHub
2. **Vercel Configuration**: Set `GROQ_API_KEY` environment variable
3. **Automatic Deployment**: Vercel handles build and deployment
4. **Verification**: Test `/health` endpoint for system status



import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

# Google SDK imports
from google.cloud import aiplatform
from google.oauth2 import service_account
import google.generativeai as genai

# Additional utilities
import sympy
import numpy as np
import requests
from bs4 import BeautifulSoup
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    MATH = "math"
    WEBSEARCH = "websearch"

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COORDINATION = "coordination"
    ERROR = "error"

@dataclass
class A2AMessage:
    """Standard message format for agent-to-agent communication"""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: str
    priority: int = 1

@dataclass
class AgentState:
    """State management for the agent system"""
    messages: List[A2AMessage]
    current_task: Optional[str]
    task_result: Optional[Dict[str, Any]]
    active_agents: List[str]
    error_log: List[str]
    conversation_history: List[Dict[str, Any]]

class GoogleSDKManager:
    """Manages Google SDK integrations"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.credentials = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients"""
        try:
            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Initialize Generative AI
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            
            logger.info("Google SDK clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google SDK: {str(e)}")
    
    async def generate_response(self, prompt: str, model_name: str = "gemini-pro") -> str:
        """Generate response using Google Generative AI"""
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

class MathAgent:
    """Specialized agent for mathematical computations"""
    
    def __init__(self, agent_id: str, google_sdk: GoogleSDKManager):
        self.agent_id = agent_id
        self.google_sdk = google_sdk
        self.capabilities = [
            "algebra", "calculus", "statistics", "geometry", 
            "number_theory", "linear_algebra", "differential_equations"
        ]
    
    async def solve_math_problem(self, problem: str) -> Dict[str, Any]:
        """Solve mathematical problems using multiple approaches"""
        try:
            result = {
                "problem": problem,
                "solution": None,
                "method": None,
                "steps": [],
                "verification": None
            }
            
            # Try symbolic computation first
            try:
                # Parse and solve using SymPy
                if any(op in problem.lower() for op in ["solve", "equation", "=", "x"]):
                    result.update(await self._solve_equation(problem))
                elif any(op in problem.lower() for op in ["derivative", "differentiate", "d/dx"]):
                    result.update(await self._solve_derivative(problem))
                elif any(op in problem.lower() for op in ["integral", "integrate"]):
                    result.update(await self._solve_integral(problem))
                elif any(op in problem.lower() for op in ["matrix", "determinant", "eigenvalue"]):
                    result.update(await self._solve_matrix(problem))
                else:
                    # Use Google AI for complex word problems
                    result.update(await self._solve_with_ai(problem))
            except Exception as e:
                logger.warning(f"Symbolic computation failed: {str(e)}")
                result.update(await self._solve_with_ai(problem))
            
            return result
            
        except Exception as e:
            logger.error(f"Math agent error: {str(e)}")
            return {"error": str(e), "problem": problem}
    
    async def _solve_equation(self, problem: str) -> Dict[str, Any]:
        """Solve algebraic equations"""
        try:
            # Extract equation from text
            equation_match = re.search(r'([^=]+=[^=]+)', problem)
            if not equation_match:
                raise ValueError("No equation found in problem")
            
            equation = equation_match.group(1)
            x = sympy.Symbol('x')
            
            # Parse equation
            left, right = equation.split('=')
            eq = sympy.Eq(sympy.sympify(left.strip()), sympy.sympify(right.strip()))
            
            # Solve
            solution = sympy.solve(eq, x)
            
            return {
                "solution": str(solution),
                "method": "symbolic_algebra",
                "steps": [f"Original equation: {equation}", f"Solution: {solution}"],
                "verification": str(sympy.simplify(eq.subs(x, solution[0]))) if solution else None
            }
        except Exception as e:
            raise Exception(f"Equation solving failed: {str(e)}")
    
    async def _solve_derivative(self, problem: str) -> Dict[str, Any]:
        """Solve derivative problems"""
        try:
            # Extract function from text
            func_match = re.search(r'f\(x\)\s*=\s*([^,\n]+)', problem)
            if not func_match:
                func_match = re.search(r'derivative of\s+([^,\n]+)', problem)
            
            if not func_match:
                raise ValueError("No function found in problem")
            
            func_str = func_match.group(1).strip()
            x = sympy.Symbol('x')
            func = sympy.sympify(func_str)
            
            derivative = sympy.diff(func, x)
            
            return {
                "solution": str(derivative),
                "method": "symbolic_differentiation",
                "steps": [f"Function: f(x) = {func_str}", f"Derivative: f'(x) = {derivative}"]
            }
        except Exception as e:
            raise Exception(f"Derivative solving failed: {str(e)}")
    
    async def _solve_integral(self, problem: str) -> Dict[str, Any]:
        """Solve integral problems"""
        try:
            # Extract function from text
            func_match = re.search(r'integral of\s+([^,\n]+)', problem)
            if not func_match:
                func_match = re.search(r'∫\s*([^,\n]+)', problem)
            
            if not func_match:
                raise ValueError("No function found in problem")
            
            func_str = func_match.group(1).strip()
            x = sympy.Symbol('x')
            func = sympy.sympify(func_str)
            
            integral = sympy.integrate(func, x)
            
            return {
                "solution": str(integral),
                "method": "symbolic_integration",
                "steps": [f"Function: f(x) = {func_str}", f"Integral: ∫f(x)dx = {integral}"]
            }
        except Exception as e:
            raise Exception(f"Integration failed: {str(e)}")
    
    async def _solve_matrix(self, problem: str) -> Dict[str, Any]:
        """Solve matrix problems"""
        try:
            # This is a simplified matrix solver
            # In production, you'd want more sophisticated parsing
            matrix_match = re.search(r'\[\[([^\]]+)\]\]', problem)
            if not matrix_match:
                raise ValueError("No matrix found in problem")
            
            # Parse matrix (simplified example)
            matrix_str = matrix_match.group(1)
            # Convert to numpy array for basic operations
            matrix = np.array(eval(f"[{matrix_str}]"))
            
            result = {}
            if "determinant" in problem.lower():
                result["solution"] = str(np.linalg.det(matrix))
                result["method"] = "determinant_calculation"
            elif "eigenvalue" in problem.lower():
                eigenvals = np.linalg.eigvals(matrix)
                result["solution"] = str(eigenvals.tolist())
                result["method"] = "eigenvalue_calculation"
            
            return result
        except Exception as e:
            raise Exception(f"Matrix solving failed: {str(e)}")
    
    async def _solve_with_ai(self, problem: str) -> Dict[str, Any]:
        """Use Google AI for complex math problems"""
        prompt = f"""
        Solve this mathematical problem step by step:
        {problem}
        
        Please provide:
        1. The solution
        2. Step-by-step working
        3. Method used
        4. Verification if possible
        
        Format your response as a clear mathematical solution.
        """
        
        response = await self.google_sdk.generate_response(prompt)
        
        return {
            "solution": response,
            "method": "ai_assisted",
            "steps": [f"AI Solution: {response}"]
        }

class WebSearchAgent:
    """Specialized agent for web search operations"""
    
    def __init__(self, agent_id: str, google_sdk: GoogleSDKManager):
        self.agent_id = agent_id
        self.google_sdk = google_sdk
        self.search_apis = {
            "google": "https://www.googleapis.com/customsearch/v1",
            "bing": "https://api.bing.microsoft.com/v7.0/search"
        }
    
    async def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Perform web search and return structured results"""
        try:
            results = {
                "query": query,
                "results": [],
                "summary": None,
                "sources": []
            }
            
            # Try Google Custom Search first
            search_results = await self._google_search(query, max_results)
            
            if not search_results:
                # Fallback to basic web scraping
                search_results = await self._basic_web_search(query, max_results)
            
            results["results"] = search_results
            results["sources"] = [r.get("link", "") for r in search_results]
            
            # Generate summary using Google AI
            if search_results:
                summary_prompt = f"""
                Based on these search results for "{query}":
                {json.dumps(search_results, indent=2)}
                
                Provide a concise summary of the key information found.
                """
                results["summary"] = await self.google_sdk.generate_response(summary_prompt)
            
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return {"error": str(e), "query": query}
    
    async def _google_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform Google Custom Search"""
        try:
            api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
            search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
            
            if not api_key or not search_engine_id:
                logger.warning("Google Custom Search credentials not found")
                return []
            
            params = {
                "key": api_key,
                "cx": search_engine_id,
                "q": query,
                "num": max_results
            }
            
            response = requests.get(self.search_apis["google"], params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google_custom_search"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Google search failed: {str(e)}")
            return []
    
    async def _basic_web_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Basic web search using DuckDuckGo (fallback)"""
        try:
            # This is a simplified implementation
            # In production, you'd use proper search APIs
            search_url = f"https://duckduckgo.com/html/?q={query}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Parse DuckDuckGo results
            for result in soup.find_all('div', class_='result')[:max_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    results.append({
                        "title": title_elem.text.strip(),
                        "link": title_elem.get('href', ''),
                        "snippet": snippet_elem.text.strip() if snippet_elem else "",
                        "source": "duckduckgo"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Basic web search failed: {str(e)}")
            return []

class AgentOrchestrator:
    """Main orchestrator managing all sub-agents"""
    
    def __init__(self, project_id: str):
        self.agent_id = "orchestrator"
        self.google_sdk = GoogleSDKManager(project_id)
        self.math_agent = MathAgent("math_agent", self.google_sdk)
        self.websearch_agent = WebSearchAgent("websearch_agent", self.google_sdk)
        
        self.agents = {
            AgentType.MATH: self.math_agent,
            AgentType.WEBSEARCH: self.websearch_agent
        }
        
        self.message_queue = asyncio.Queue()
        self.active_tasks = {}
    
    async def process_request(self, user_query: str) -> Dict[str, Any]:
        """Main entry point for processing user requests"""
        try:
            # Analyze query to determine routing
            task_analysis = await self._analyze_query(user_query)
            
            # Route to appropriate agent(s)
            if task_analysis["requires_multiple_agents"]:
                result = await self._coordinate_multi_agent_task(user_query, task_analysis)
            else:
                result = await self._single_agent_task(user_query, task_analysis)
            
            return {
                "query": user_query,
                "result": result,
                "agents_used": task_analysis["agents_needed"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Orchestrator error: {str(e)}")
            return {"error": str(e), "query": user_query}
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine routing strategy"""
        analysis_prompt = f"""
        Analyze this user query and determine which agents are needed:
        Query: "{query}"
        
        Available agents:
        1. Math Agent - handles mathematical computations, equations, calculus, statistics
        2. Web Search Agent - handles information retrieval, current events, factual queries
        
        Respond with JSON format:
        {{
            "agents_needed": ["math", "websearch"],
            "requires_multiple_agents": true/false,
            "primary_agent": "math" or "websearch",
            "reasoning": "explanation of routing decision"
        }}
        """
        
        response = await self.google_sdk.generate_response(analysis_prompt)
        
        try:
            # Parse AI response
            import json
            analysis = json.loads(response)
        except:
            # Fallback analysis
            analysis = {
                "agents_needed": [],
                "requires_multiple_agents": False,
                "primary_agent": None,
                "reasoning": "Fallback analysis"
            }
            
            # Simple keyword-based routing
            math_keywords = ["solve", "equation", "calculate", "math", "derivative", "integral", "matrix"]
            search_keywords = ["search", "find", "what is", "current", "news", "information"]
            
            if any(keyword in query.lower() for keyword in math_keywords):
                analysis["agents_needed"].append("math")
                analysis["primary_agent"] = "math"
            
            if any(keyword in query.lower() for keyword in search_keywords):
                analysis["agents_needed"].append("websearch")
                if not analysis["primary_agent"]:
                    analysis["primary_agent"] = "websearch"
            
            analysis["requires_multiple_agents"] = len(analysis["agents_needed"]) > 1
        
        return analysis
    
    async def _single_agent_task(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle single agent tasks"""
        primary_agent = analysis["primary_agent"]
        
        if primary_agent == "math":
            return await self.math_agent.solve_math_problem(query)
        elif primary_agent == "websearch":
            return await self.websearch_agent.search_web(query)
        else:
            return {"error": "No suitable agent found for query"}
    
    async def _coordinate_multi_agent_task(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multi-agent tasks"""
        results = {}
        
        # Execute tasks in parallel
        tasks = []
        
        if "math" in analysis["agents_needed"]:
            tasks.append(self.math_agent.solve_math_problem(query))
        
        if "websearch" in analysis["agents_needed"]:
            tasks.append(self.websearch_agent.search_web(query))
        
        # Wait for all tasks to complete
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for i, agent_type in enumerate(analysis["agents_needed"]):
            if isinstance(task_results[i], Exception):
                results[agent_type] = {"error": str(task_results[i])}
            else:
                results[agent_type] = task_results[i]
        
        # Generate final synthesis
        synthesis_prompt = f"""
        Synthesize these results from multiple agents for the query: "{query}"
        
        Results:
        {json.dumps(results, indent=2)}
        
        Provide a coherent, integrated response that combines the information from all agents.
        """
        
        synthesis = await self.google_sdk.generate_response(synthesis_prompt)
        
        return {
            "individual_results": results,
            "synthesis": synthesis,
            "coordination_method": "parallel_execution"
        }

class A2AProtocolSystem:
    """Main A2A Protocol System"""
    
    def __init__(self, project_id: str):
        self.orchestrator = AgentOrchestrator(project_id)
        self.system_state = AgentState(
            messages=[],
            current_task=None,
            task_result=None,
            active_agents=[],
            error_log=[],
            conversation_history=[]
        )
    
    async def process_user_query(self, query: str) -> Dict[str, Any]:
        """Main interface for processing user queries"""
        try:
            logger.info(f"Processing query: {query}")
            
            # Update system state
            self.system_state.current_task = query
            self.system_state.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "user_query",
                "content": query
            })
            
            # Process through orchestrator
            result = await self.orchestrator.process_request(query)
            
            # Update system state
            self.system_state.task_result = result
            self.system_state.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "system_response",
                "content": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"System error: {str(e)}")
            error_result = {"error": str(e), "query": query}
            self.system_state.error_log.append(error_result)
            return error_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "active_agents": ["orchestrator", "math_agent", "websearch_agent"],
            "total_conversations": len(self.system_state.conversation_history),
            "error_count": len(self.system_state.error_log),
            "current_task": self.system_state.current_task,
            "last_activity": datetime.now().isoformat()
        }

# Example usage and testing
async def main():
    """Example usage of the A2A Protocol System"""
    
    # Initialize system
    system = A2AProtocolSystem(project_id="your-google-project-id")
    
    # Test queries
    test_queries = [
        "Solve the equation 2x + 5 = 15",
        "What is the current weather in New York?",
        "Calculate the derivative of x^2 + 3x + 2 and find information about derivatives",
        "What are the eigenvalues of the matrix [[1, 2], [3, 4]]?",
        "Search for recent news about artificial intelligence"
    ]
    
    print("A2A Protocol System Demo")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        result = await system.process_user_query(query)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Result: {json.dumps(result, indent=2)}")
    
    # Show system status
    print("\nSystem Status:")
    print(json.dumps(system.get_system_status(), indent=2))

if __name__ == "__main__":
    # Set up environment variables
    os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
    os.environ["GOOGLE_SEARCH_API_KEY"] = "your-google-search-api-key"
    os.environ["GOOGLE_SEARCH_ENGINE_ID"] = "your-search-engine-id"
    
    # Run the demo
    asyncio.run(main())
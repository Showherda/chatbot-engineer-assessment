# tests/test_planner_new.py
import pytest
import asyncio
from main import ChatbotPlanner


class TestPlanner:
    """Test the planner's intent classification and action selection"""

    def test_arithmetic_detection(self):
        """Test arithmetic intent detection"""
        test_cases = [
            "2 + 3",
            "10 * 5 - 2",
            "(4 + 6) / 2",
            "calculate 5 + 7",
            "what is 3 * 4?",
            "compute 100 / 5",
        ]

        for case in test_cases:
            intent = ChatbotPlanner.parse_intent(case)
            assert intent["intent"] == "calculate"
            assert intent["confidence"] >= 0.8

    def test_product_search_detection(self):
        """Test product search intent detection"""
        test_cases = [
            "show me cups",
            "I want to buy a tumbler",
            "what mugs do you have?",
            "show me products",
            "find me a bottle",
            "product search",
        ]

        for case in test_cases:
            intent = ChatbotPlanner.parse_intent(case)
            assert intent["intent"] == "search_products"
            assert intent["confidence"] >= 0.7

    def test_outlet_search_detection(self):
        """Test outlet search intent detection"""
        test_cases = [
            "where is the nearest store?",
            "find outlets in KL",
            "show me branches",
            "outlet locations",
            "where can I find ZUS Coffee?",
            "store addresses",
        ]

        for case in test_cases:
            intent = ChatbotPlanner.parse_intent(case)
            assert intent["intent"] == "search_outlets"
            assert intent["confidence"] >= 0.7

    def test_general_chat_detection(self):
        """Test general chat intent detection"""
        test_cases = [
            "hello",
            "how are you?",
            "tell me about ZUS Coffee",
            "what's the weather like?",
            "good morning",
        ]

        for case in test_cases:
            intent = ChatbotPlanner.parse_intent(case)
            assert intent["intent"] == "general_chat"

    def test_action_planning(self):
        """Test action planning from intent"""
        # Test calculate action
        intent = {"intent": "calculate", "expression": "2 + 3", "confidence": 0.9}
        action = ChatbotPlanner.plan_action(intent, "user123")
        assert action["action"] == "calculate"
        assert action["parameters"]["expression"] == "2 + 3"

        # Test product search action
        intent = {
            "intent": "search_products",
            "query": "show me cups",
            "confidence": 0.8,
        }
        action = ChatbotPlanner.plan_action(intent, "user123")
        assert action["action"] == "search_products"
        assert action["parameters"]["query"] == "show me cups"

        # Test outlet search action
        intent = {
            "intent": "search_outlets",
            "query": "find stores in KL",
            "confidence": 0.8,
        }
        action = ChatbotPlanner.plan_action(intent, "user123")
        assert action["action"] == "search_outlets"
        assert action["parameters"]["query"] == "find stores in KL"

        # Test general response action
        intent = {"intent": "general_chat", "query": "hello", "confidence": 0.5}
        action = ChatbotPlanner.plan_action(intent, "user123")
        assert action["action"] == "general_response"
        assert action["parameters"]["message"] == "hello"

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty input
        intent = ChatbotPlanner.parse_intent("")
        assert intent["intent"] == "general_chat"

        # Very long input
        long_input = "a" * 1000
        intent = ChatbotPlanner.parse_intent(long_input)
        assert intent["intent"] == "general_chat"

        # Mixed signals
        mixed_input = "calculate 2 + 3 and show me cups"
        intent = ChatbotPlanner.parse_intent(mixed_input)
        # Should pick the first detected intent (calculate)
        assert intent["intent"] == "calculate"

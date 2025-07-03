# tests/test_calculator_new.py
import pytest
from main import Calculator


class TestCalculator:
    """Test calculator functionality"""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations"""
        result = Calculator.evaluate("2 + 3")
        assert result["success"] == True
        assert result["result"] == 5

        result = Calculator.evaluate("10 - 4")
        assert result["success"] == True
        assert result["result"] == 6

        result = Calculator.evaluate("3 * 7")
        assert result["success"] == True
        assert result["result"] == 21

        result = Calculator.evaluate("15 / 3")
        assert result["success"] == True
        assert result["result"] == 5

    def test_complex_expressions(self):
        """Test complex arithmetic expressions"""
        result = Calculator.evaluate("2 + 3 * 4")
        assert result["success"] == True
        assert result["result"] == 14  # Order of operations

        result = Calculator.evaluate("(2 + 3) * 4")
        assert result["success"] == True
        assert result["result"] == 20  # Parentheses

        result = Calculator.evaluate("10 / 2 + 3")
        assert result["success"] == True
        assert result["result"] == 8

    def test_decimal_numbers(self):
        """Test decimal number calculations"""
        result = Calculator.evaluate("2.5 + 1.5")
        assert result["success"] == True
        assert result["result"] == 4.0

        result = Calculator.evaluate("10.5 / 2")
        assert result["success"] == True
        assert result["result"] == 5.25

    def test_invalid_expressions(self):
        """Test handling of invalid expressions"""
        result = Calculator.evaluate("2 + abc")
        assert result["success"] == False
        assert "error" in result

    def test_security_protection(self):
        """Test security against malicious inputs"""
        result = Calculator.evaluate("import os")
        assert result["success"] == False
        assert "error" in result

    def test_empty_input(self):
        """Test handling of empty input"""
        result = Calculator.evaluate("")
        assert result["success"] == False
        assert "error" in result

        result = Calculator.evaluate("   ")
        assert result["success"] == False
        assert "error" in result

    def test_division_by_zero(self):
        """Test division by zero handling"""
        result = Calculator.evaluate("5 / 0")
        assert result["success"] == False
        assert "error" in result

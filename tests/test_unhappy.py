# tests/test_unhappy.py
import pytest
import asyncio
from httpx import AsyncClient
from main import app
from unittest.mock import patch


class TestUnhappyFlows:
    """Test robustness against invalid inputs and error conditions"""

    @pytest.mark.asyncio
    async def test_missing_parameters_calculator(self):
        """Test calculator with missing or invalid parameters"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            test_cases = [
                "Calculate",  # No expression
                "What is",  # Incomplete
                "Compute something",  # Vague
                "Math please",  # No numbers
            ]

            for case in test_cases:
                response = await client.post("/chat", json={"message": case})
                assert response.status_code == 200
                data = response.json()
                # Should handle gracefully without crashing
                assert "response" in data
                # Should indicate inability to process
                assert any(
                    word in data["response"].lower()
                    for word in ["sorry", "couldn't", "understand", "try"]
                )

    @pytest.mark.asyncio
    async def test_missing_parameters_outlets(self):
        """Test outlets query with missing parameters"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            test_cases = [
                "Show outlets",  # Too vague
                "Find store",  # No location
                "Opening time",  # No specific outlet
                "Where",  # Incomplete
            ]

            for case in test_cases:
                response = await client.post("/chat", json={"message": case})
                assert response.status_code == 200
                data = response.json()
                # Should handle gracefully
                assert "response" in data

    @pytest.mark.asyncio
    async def test_missing_parameters_products(self):
        """Test products query with missing parameters"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            test_cases = [
                "Show products",  # Too vague
                "Buy something",  # No specific product
                "Price",  # No product specified
                "What do you sell",  # Very general
            ]

            for case in test_cases:
                response = await client.post("/chat", json={"message": case})
                assert response.status_code == 200
                data = response.json()
                assert "response" in data

    @pytest.mark.asyncio
    async def test_api_downtime_simulation(self):
        """Test handling of API downtime"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Mock API client to simulate downtime
            with patch(
                "app.integrations.api_client.APIClient.query_products"
            ) as mock_products:
                mock_products.side_effect = Exception("Service temporarily unavailable")

                response = await client.post(
                    "/chat", json={"message": "Show me coffee mugs"}
                )

                assert response.status_code == 200
                data = response.json()
                # Should handle the error gracefully
                assert (
                    "trouble" in data["response"].lower()
                    or "try again" in data["response"].lower()
                )

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self):
        """Test SQL injection prevention in outlets API"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            malicious_queries = [
                "outlets'; DROP TABLE outlets; --",
                "location UNION SELECT * FROM outlets",
                "'; DELETE FROM outlets WHERE '1'='1",
                "outlets' OR '1'='1",
                "location'; INSERT INTO outlets VALUES ('evil', 'hack'); --",
            ]

            for malicious_query in malicious_queries:
                response = await client.get(
                    "/outlets", params={"query": malicious_query}
                )

                # Should either reject the query or handle it safely
                if response.status_code == 400:
                    # Rejected - good
                    data = response.json()
                    assert (
                        "error" in data["detail"].lower()
                        or "dangerous" in data["detail"].lower()
                    )
                elif response.status_code == 200:
                    # Handled safely - database should be intact
                    # Check that no malicious operations occurred
                    data = response.json()
                    assert "results" in data

    @pytest.mark.asyncio
    async def test_calculator_malicious_input(self):
        """Test calculator with malicious input"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            malicious_expressions = [
                "import os; os.system('rm -rf /')",
                "__import__('os').system('ls')",
                "exec('print(1)')",
                "eval('2+2')",
                "1**10**10**10",  # Extremely large number
                "9" * 1000,  # Very long number
            ]

            for expr in malicious_expressions:
                response = await client.post("/chat", json={"message": expr})
                assert response.status_code == 200
                data = response.json()
                # Should either safely calculate or reject
                assert "response" in data
                # Should not execute arbitrary code

    @pytest.mark.asyncio
    async def test_extremely_long_input(self):
        """Test handling of extremely long input"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            long_message = "A" * 10000  # Very long message

            response = await client.post("/chat", json={"message": long_message})
            assert response.status_code in [200, 413, 422]  # Should handle gracefully

            if response.status_code == 200:
                data = response.json()
                assert "response" in data

    @pytest.mark.asyncio
    async def test_invalid_json_payload(self):
        """Test handling of invalid JSON payloads"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test various malformed requests
            response = await client.post("/chat", content="invalid json")
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_message_handling(self):
        """Test handling of empty messages"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            test_cases = [
                "",  # Empty string
                "   ",  # Whitespace only
                "\n\t\r",  # Various whitespace
            ]

            for case in test_cases:
                response = await client.post("/chat", json={"message": case})
                assert response.status_code == 200
                data = response.json()
                assert "response" in data
                # Should handle empty input gracefully

    @pytest.mark.asyncio
    async def test_special_characters_handling(self):
        """Test handling of special characters and unicode"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            special_inputs = [
                "🦀🦀🦀 coffee emoji test",
                "café with accénts",
                "中文 Chinese characters",
                "🏪 Where is the store? 📍",
                "SELECT * FROM outlets 💀",
                "\x00\x01\x02 null bytes",
            ]

            for special_input in special_inputs:
                response = await client.post("/chat", json={"message": special_input})
                assert response.status_code == 200
                data = response.json()
                assert "response" in data
                # Should handle special characters without crashing

    @pytest.mark.asyncio
    async def test_concurrent_requests_stress(self):
        """Test handling of concurrent requests"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Send multiple concurrent requests
            tasks = []
            for i in range(10):
                task = client.post("/chat", json={"message": f"Test message {i}"})
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # All should complete without crashing
            for response in responses:
                if isinstance(response, Exception):
                    pytest.fail(f"Request failed with exception: {response}")
                assert response.status_code == 200

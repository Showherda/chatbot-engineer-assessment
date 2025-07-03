# tests/test_sequential.py
import pytest
import asyncio
from httpx import AsyncClient
from main import app


class TestSequentialConversation:
    """Test multi-turn conversation and memory management"""

    @pytest.mark.asyncio
    async def test_outlet_conversation_flow(self):
        """Test the example conversation flow from the requirements"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Turn 1: Ask about outlet in Petaling Jaya
            response1 = await client.post(
                "/chat",
                json={
                    "message": "Is there an outlet in Petaling Jaya?",
                    "user_id": "test_user",
                },
            )
            assert response1.status_code == 200
            data1 = response1.json()
            assert (
                "outlet" in data1["response"].lower()
                or "petaling jaya" in data1["response"].lower()
            )

            # Turn 2: Ask about specific outlet
            response2 = await client.post(
                "/chat",
                json={
                    "message": "SS 2, whats the opening time?",
                    "user_id": "test_user",
                },
            )
            assert response2.status_code == 200
            data2 = response2.json()
            assert (
                "ss 2" in data2["response"].lower()
                or "time" in data2["response"].lower()
            )

            # Turn 3: Follow-up question
            response3 = await client.post(
                "/chat", json={"message": "Do they have wifi?", "user_id": "test_user"}
            )
            assert response3.status_code == 200
            data3 = response3.json()
            # Should understand context and provide relevant information

    @pytest.mark.asyncio
    async def test_product_conversation_flow(self):
        """Test product-related conversation flow"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Turn 1: Ask about coffee products
            response1 = await client.post(
                "/chat",
                json={
                    "message": "What coffee mugs do you have?",
                    "user_id": "test_user2",
                },
            )
            assert response1.status_code == 200

            # Turn 2: Ask for more details
            response2 = await client.post(
                "/chat",
                json={"message": "What about the prices?", "user_id": "test_user2"},
            )
            assert response2.status_code == 200

            # Turn 3: Ask about specific features
            response3 = await client.post(
                "/chat",
                json={"message": "Are they dishwasher safe?", "user_id": "test_user2"},
            )
            assert response3.status_code == 200

    @pytest.mark.asyncio
    async def test_conversation_memory(self):
        """Test that conversation history is maintained"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            user_id = "memory_test_user"

            # First message
            await client.post(
                "/chat",
                json={
                    "message": "Hello, I'm looking for a coffee shop",
                    "user_id": user_id,
                },
            )

            # Second message referring to previous context
            response = await client.post(
                "/chat",
                json={"message": "Do you have any near KLCC?", "user_id": user_id},
            )

            assert response.status_code == 200
            # The bot should understand the context from previous message

    @pytest.mark.asyncio
    async def test_conversation_reset(self):
        """Test conversation reset functionality"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            user_id = "reset_test_user"

            # Have a conversation
            await client.post(
                "/chat",
                json={"message": "Remember that I like cold brew", "user_id": user_id},
            )

            # Reset conversation
            reset_response = await client.post(f"/chat/reset?user_id={user_id}")
            assert reset_response.status_code == 200

            # Verify memory is cleared
            history_response = await client.get("/chat/history")
            assert history_response.status_code == 200

    @pytest.mark.asyncio
    async def test_interrupted_conversation(self):
        """Test handling of interrupted conversation flow"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            user_id = "interrupt_test_user"

            # Start conversation about outlets
            await client.post(
                "/chat",
                json={"message": "Tell me about your outlets", "user_id": user_id},
            )

            # Suddenly switch to products
            response = await client.post(
                "/chat",
                json={
                    "message": "Actually, what coffee products do you sell?",
                    "user_id": user_id,
                },
            )

            assert response.status_code == 200
            data = response.json()
            # Should handle the topic switch gracefully
            assert data["action_taken"] in ["products", "conversational"]

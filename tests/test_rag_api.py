# tests/test_rag_api.py
import pytest
import asyncio
from httpx import AsyncClient
from main import app


class TestProductsRAGAPI:
    """Test the Products RAG API functionality"""

    @pytest.mark.asyncio
    async def test_products_health_check(self):
        """Test products API health endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data

    @pytest.mark.asyncio
    async def test_products_query_get(self):
        """Test products query using GET method"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/products", params={"query": "coffee mug"})

            if response.status_code == 503:
                # Vector store not available - this is expected in test environment
                pytest.skip("Vector store not available in test environment")

            assert response.status_code == 200
            data = response.json()
            assert "summary" in data
            assert "relevant_products" in data
            assert "query" in data

    @pytest.mark.asyncio
    async def test_products_query_validation(self):
        """Test products query parameter validation"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Missing query parameter
            response = await client.get("/products")
            assert response.status_code == 422

            # Invalid top_k parameter
            response = await client.get(
                "/products", params={"query": "test", "top_k": 0}
            )
            assert response.status_code == 422


class TestOutletsText2SQLAPI:
    """Test the Outlets Text2SQL API functionality"""

    @pytest.mark.asyncio
    async def test_outlets_health_check(self):
        """Test outlets API health endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data

    @pytest.mark.asyncio
    async def test_outlets_query_get(self):
        """Test outlets query using GET method"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/outlets", params={"query": "outlets in Petaling Jaya"}
            )
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "sql_query" in data
            assert "summary" in data
            assert "natural_language_query" in data

    @pytest.mark.asyncio
    async def test_outlets_all_endpoint(self):
        """Test get all outlets endpoint - using general query"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Since unified app doesn't have /outlets/all, test general outlets query
            response = await client.get("/outlets", params={"query": "all outlets"})
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "sql_query" in data

    @pytest.mark.asyncio
    async def test_outlets_query_validation(self):
        """Test outlets query parameter validation"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Missing query parameter
            response = await client.get("/outlets")
            assert response.status_code == 422


class TestAPIIntegration:
    """Test integration between main app and external APIs"""

    @pytest.mark.asyncio
    async def test_successful_product_integration(self):
        """Test successful integration with products API"""
        # This would test the actual integration, but requires running services
        # In a real test environment, you'd use test containers or mock services
        pass

    @pytest.mark.asyncio
    async def test_successful_outlet_integration(self):
        """Test successful integration with outlets API"""
        # This would test the actual integration with real services
        pass

    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test handling of API errors"""
        # Test timeout, connection errors, HTTP errors
        pass

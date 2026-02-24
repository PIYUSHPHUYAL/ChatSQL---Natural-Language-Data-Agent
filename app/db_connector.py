import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Handles PostgreSQL database connections safely."""

    def __init__(self):
        """Initialize database connector with environment variables."""
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = os.getenv('DB_PORT', '5432')
        self.database = os.getenv('DB_NAME', 'testdb')
        self.user = os.getenv('DB_USER', 'dbuser')
        self.password = os.getenv('DB_PASSWORD', 'dbpass123')
        self.engine = None
        self.connection_string = None

    def build_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def connect(self) -> bool:
        """
        Establish connection to PostgreSQL database.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection_string = self.build_connection_string()
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,  # Verify connections before using
                pool_size=5,
                max_overflow=10
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info(f"✅ Connected to database: {self.database}")
            return True

        except SQLAlchemyError as e:
            logger.error(f"❌ Database connection failed: {str(e)}")
            return False

    def test_connection(self) -> Dict[str, Any]:
        """
        Test database connection and return status.

        Returns:
            dict: Connection status and details
        """
        try:
            if not self.engine:
                return {
                    "connected": False,
                    "error": "No connection established. Call connect() first."
                }

            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]

                return {
                    "connected": True,
                    "database": self.database,
                    "host": self.host,
                    "version": version.split('\n')[0]  # Just first line
                }

        except SQLAlchemyError as e:
            return {
                "connected": False,
                "error": str(e)
            }

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a SQL query safely (read-only).

        Args:
            query: SQL query string
            params: Optional parameters for query

        Returns:
            list: Query results as list of dictionaries
        """
        try:
            # Safety check - only allow SELECT statements
            query_upper = query.strip().upper()
            if not query_upper.startswith('SELECT'):
                raise ValueError("Only SELECT queries are allowed. No INSERT, UPDATE, DELETE, or DROP.")

            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})

                # Convert to list of dicts
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]

                logger.info(f"✅ Query executed successfully. Returned {len(rows)} rows.")
                return rows

        except ValueError as e:
            logger.error(f"❌ Security violation: {str(e)}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"❌ Query execution failed: {str(e)}")
            raise

    def get_table_names(self) -> List[str]:
        """
        Get list of all tables in the database.

        Returns:
            list: Table names
        """
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """
        try:
            results = self.execute_query(query)
            return [row['table_name'] for row in results]
        except Exception as e:
            logger.error(f"❌ Failed to get table names: {str(e)}")
            return []

    def get_row_count(self, table_name: str) -> int:
        """
        Get number of rows in a table.

        Args:
            table_name: Name of the table

        Returns:
            int: Number of rows
        """
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        try:
            result = self.execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"❌ Failed to count rows in {table_name}: {str(e)}")
            return 0

    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("✅ Database connection closed.")


# Singleton instance
_db_connector = None

def get_db_connector() -> DatabaseConnector:
    """Get or create database connector singleton."""
    global _db_connector
    if _db_connector is None:
        _db_connector = DatabaseConnector()
    return _db_connector
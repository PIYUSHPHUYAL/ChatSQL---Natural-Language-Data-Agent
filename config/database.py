"""
Database configuration for ChatSQL
Connects to existing PostgreSQL whale tracker database
"""

import psycopg2
from typing import Optional


class DatabaseConfig:
    """Database connection configuration"""

    # Your whale database credentials
    HOST = "localhost"
    PORT = 5432
    DATABASE = "crypto_tracker"
    USER = "crypto_admin"
    PASSWORD = "crypto_secure_pass_123"

    @classmethod
    def get_connection_string(cls) -> str:
        """Returns PostgreSQL connection string"""
        return f"postgresql://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.DATABASE}"

    @classmethod
    def test_connection(cls) -> bool:
        """Test if we can connect to database"""
        try:
            conn = psycopg2.connect(
                host=cls.HOST,
                port=cls.PORT,
                database=cls.DATABASE,
                user=cls.USER,
                password=cls.PASSWORD
            )
            conn.close()
            print(f"✅ Connected to database: {cls.DATABASE}")
            return True
        except psycopg2.OperationalError as e:
            print(f"❌ Cannot connect to database: {e}")
            return False


def test_db_connection():
    """Quick test of database connection"""
    print("Testing PostgreSQL connection...")

    if DatabaseConfig.test_connection():
        # Try a simple query
        conn = psycopg2.connect(
            host=DatabaseConfig.HOST,
            port=DatabaseConfig.PORT,
            database=DatabaseConfig.DATABASE,
            user=DatabaseConfig.USER,
            password=DatabaseConfig.PASSWORD
        )

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM alerts;")
        count = cursor.fetchone()[0]

        print(f"✅ Found {count} whale alerts in database!")

        cursor.close()
        conn.close()
    else:
        print("❌ Fix database connection before continuing")


if __name__ == "__main__":
    test_db_connection()
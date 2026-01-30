"""
Schema Loader - Extracts database schema from PostgreSQL
Builds structured metadata for semantic search
"""

import psycopg2
from typing import List, Dict, Any
import json


class SchemaLoader:
    """
    Extracts database schema information from PostgreSQL.
    Creates semantic descriptions for tables and columns.
    """

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        """Initialize with database credentials"""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None

    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        print(f"✅ Connected to {self.database}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_all_tables(self) -> List[str]:
        """Get list of all tables in public schema"""
        cursor = self.conn.cursor()

        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """

        cursor.execute(query)
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()

        return tables

    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get detailed column information for a table.

        Returns list of dicts with: column_name, data_type, is_nullable
        """
        cursor = self.conn.cursor()

        query = """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name = %s
        ORDER BY ordinal_position;
        """

        cursor.execute(query, (table_name,))

        columns = []
        for row in cursor.fetchall():
            columns.append({
                'name': row[0],
                'type': row[1],
                'nullable': row[2] == 'YES',
                'default': row[3]
            })

        cursor.close()
        return columns

    def get_sample_data(self, table_name: str, limit: int = 3) -> List[tuple]:
        """Get sample rows from table (for understanding data)"""
        cursor = self.conn.cursor()

        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit};"
            cursor.execute(query)
            samples = cursor.fetchall()
            cursor.close()
            return samples
        except Exception as e:
            print(f"⚠️  Could not fetch sample data from {table_name}: {e}")
            cursor.close()
            return []

    def create_table_description(self, table_name: str) -> str:
        """
        Create a semantic description of the table.
        This will be used for vector search.
        """
        columns = self.get_table_columns(table_name)

        # Build description
        description = f"Table: {table_name}\n"
        description += f"Columns:\n"

        for col in columns:
            description += f"  - {col['name']} ({col['type']})"
            if not col['nullable']:
                description += " NOT NULL"
            description += "\n"

        # Add business context (you can customize this!)
        context = self._get_business_context(table_name)
        if context:
            description += f"\nPurpose: {context}\n"

        return description

    def _get_business_context(self, table_name: str) -> str:
        """
        Add business/domain knowledge about tables.
        This helps the LLM understand what data means.
        """
        contexts = {
            'alerts': 'Stores whale trade alerts - large cryptocurrency transactions over $500K. Includes trade details, prices, and detection timestamps.',
            'latest_prices': 'Current cryptocurrency prices for tracked symbols (BTC, ETH, BNB). Updated in real-time from market data.',
            'trade_stats': 'Aggregated trading statistics including volume, trade counts, and average prices per time window.',
            'whale_config': 'Configuration table for whale detection thresholds and parameters per symbol.'
        }

        return contexts.get(table_name, '')

    def extract_full_schema(self) -> Dict[str, Any]:
        """
        Extract complete schema for all tables.

        Returns dict with table metadata and descriptions.
        """
        tables = self.get_all_tables()

        schema_data = {
            'database': self.database,
            'tables': {}
        }

        print(f"\n📊 Extracting schema from {len(tables)} tables...\n")

        for table in tables:
            print(f"  Processing: {table}")

            columns = self.get_table_columns(table)
            description = self.create_table_description(table)
            samples = self.get_sample_data(table, limit=2)

            schema_data['tables'][table] = {
                'name': table,
                'columns': columns,
                'description': description,
                'sample_count': len(samples)
            }

        print(f"\n✅ Schema extraction complete!\n")

        return schema_data

    def save_schema(self, schema_data: Dict[str, Any], output_file: str = 'schema.json'):
        """Save extracted schema to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(schema_data, f, indent=2, default=str)

        print(f"💾 Schema saved to: {output_file}")


def test_schema_loader():
    """Test schema extraction"""
    from config.database import DatabaseConfig

    print("Testing Schema Loader...\n")

    # Create loader
    loader = SchemaLoader(
        host=DatabaseConfig.HOST,
        port=DatabaseConfig.PORT,
        database=DatabaseConfig.DATABASE,
        user=DatabaseConfig.USER,
        password=DatabaseConfig.PASSWORD
    )

    # Connect
    loader.connect()

    # Get all tables
    tables = loader.get_all_tables()
    print(f"Found {len(tables)} tables: {tables}\n")

    # Get details for alerts table
    print("=" * 60)
    print("ALERTS TABLE DETAILS:")
    print("=" * 60)

    columns = loader.get_table_columns('alerts')
    print(f"\nColumns ({len(columns)}):")
    for col in columns:
        print(f"  - {col['name']:20} {col['type']:15} {'NULL' if col['nullable'] else 'NOT NULL'}")

    # Get description
    description = loader.create_table_description('alerts')
    print(f"\n{description}")

    # Extract full schema
    print("\n" + "=" * 60)
    print("EXTRACTING FULL SCHEMA:")
    print("=" * 60)

    schema = loader.extract_full_schema()

    # Save to file
    loader.save_schema(schema, 'config/schema.json')

    # Close connection
    loader.close()

    print("\n✅ Schema loader test complete!")


if __name__ == "__main__":
    test_schema_loader()
import logging
from typing import Dict, List, Any
from sqlalchemy import text
from db_connector import DatabaseConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaDiscovery:
    """Automatically discovers and analyzes database schema."""

    def __init__(self, db_connector: DatabaseConnector):
        """Initialize with database connector."""
        self.db = db_connector
        self.schema_cache = None

    def discover_full_schema(self) -> Dict[str, Any]:
        """
        Discover complete database schema including tables, columns, and relationships.

        Returns:
            dict: Complete schema information
        """
        try:
            logger.info("🔍 Starting schema discovery...")

            schema = {
                "tables": {},
                "total_tables": 0,
                "total_columns": 0,
                "relationships": []
            }

            # Get all tables
            tables = self.db.get_table_names()
            schema["total_tables"] = len(tables)

            # For each table, get detailed info
            for table_name in tables:
                logger.info(f"   📊 Analyzing table: {table_name}")

                table_info = {
                    "columns": self._get_columns(table_name),
                    "row_count": self.db.get_row_count(table_name),
                    "primary_key": self._get_primary_key(table_name),
                    "foreign_keys": self._get_foreign_keys(table_name)
                }

                schema["tables"][table_name] = table_info
                schema["total_columns"] += len(table_info["columns"])

                # Add relationships
                for fk in table_info["foreign_keys"]:
                    schema["relationships"].append({
                        "from_table": table_name,
                        "from_column": fk["column"],
                        "to_table": fk["references_table"],
                        "to_column": fk["references_column"]
                    })

            # Cache the schema
            self.schema_cache = schema

            logger.info(f"✅ Schema discovery complete!")
            logger.info(f"   📊 Tables: {schema['total_tables']}")
            logger.info(f"   📝 Columns: {schema['total_columns']}")
            logger.info(f"   🔗 Relationships: {len(schema['relationships'])}")

            return schema

        except Exception as e:
            logger.error(f"❌ Schema discovery failed: {str(e)}")
            return {"error": str(e)}

    def _get_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a table.

        Args:
            table_name: Name of the table

        Returns:
            list: Column details
        """
        query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = :table_name
            ORDER BY ordinal_position
        """

        try:
            results = self.db.execute_query(query, {"table_name": table_name})

            columns = []
            for row in results:
                columns.append({
                    "name": row["column_name"],
                    "type": row["data_type"],
                    "nullable": row["is_nullable"] == "YES",
                    "default": row["column_default"],
                    "max_length": row["character_maximum_length"]
                })

            return columns

        except Exception as e:
            logger.error(f"❌ Failed to get columns for {table_name}: {str(e)}")
            return []

    def _get_primary_key(self, table_name: str) -> List[str]:
        """
        Get primary key column(s) for a table.

        Args:
            table_name: Name of the table

        Returns:
            list: Primary key column names
        """
        query = """
            SELECT a.attname as column_name
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid
                AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = :table_name::regclass
              AND i.indisprimary
        """

        try:
            results = self.db.execute_query(query, {"table_name": table_name})
            return [row["column_name"] for row in results]

        except Exception as e:
            logger.error(f"❌ Failed to get primary key for {table_name}: {str(e)}")
            return []

    def _get_foreign_keys(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get foreign key relationships for a table.

        Args:
            table_name: Name of the table

        Returns:
            list: Foreign key details
        """
        query = """
            SELECT
                kcu.column_name,
                ccu.table_name AS references_table,
                ccu.column_name AS references_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_name = :table_name
        """

        try:
            results = self.db.execute_query(query, {"table_name": table_name})

            foreign_keys = []
            for row in results:
                foreign_keys.append({
                    "column": row["column_name"],
                    "references_table": row["references_table"],
                    "references_column": row["references_column"]
                })

            return foreign_keys

        except Exception as e:
            logger.error(f"❌ Failed to get foreign keys for {table_name}: {str(e)}")
            return []

    def get_schema_summary(self) -> str:
        """
        Get a human-readable summary of the schema.

        Returns:
            str: Schema summary
        """
        if not self.schema_cache:
            self.discover_full_schema()

        if not self.schema_cache or "error" in self.schema_cache:
            return "❌ Schema not available"

        summary = []
        summary.append(f"📊 **Database Schema Summary**\n")
        summary.append(f"Total Tables: {self.schema_cache['total_tables']}")
        summary.append(f"Total Columns: {self.schema_cache['total_columns']}")
        summary.append(f"Relationships: {len(self.schema_cache['relationships'])}\n")

        summary.append("**Tables:**")
        for table_name, table_info in self.schema_cache["tables"].items():
            summary.append(f"  • {table_name} ({len(table_info['columns'])} columns, {table_info['row_count']} rows)")

        return "\n".join(summary)

    def get_table_details(self, table_name: str) -> str:
        """
        Get detailed information about a specific table.

        Args:
            table_name: Name of the table

        Returns:
            str: Table details
        """
        if not self.schema_cache:
            self.discover_full_schema()

        if table_name not in self.schema_cache["tables"]:
            return f"❌ Table '{table_name}' not found"

        table = self.schema_cache["tables"][table_name]

        details = []
        details.append(f"📊 **Table: {table_name}**\n")
        details.append(f"Row Count: {table['row_count']}")
        details.append(f"Primary Key: {', '.join(table['primary_key']) if table['primary_key'] else 'None'}\n")

        details.append("**Columns:**")
        for col in table["columns"]:
            nullable = "NULL" if col["nullable"] else "NOT NULL"
            details.append(f"  • {col['name']} ({col['type']}) - {nullable}")

        if table["foreign_keys"]:
            details.append("\n**Foreign Keys:**")
            for fk in table["foreign_keys"]:
                details.append(f"  • {fk['column']} → {fk['references_table']}.{fk['references_column']}")

        return "\n".join(details)

    def get_schema_for_llm(self) -> str:
        """
        Get schema formatted for LLM context (concise).

        Returns:
            str: LLM-friendly schema description
        """
        if not self.schema_cache:
            self.discover_full_schema()

        if not self.schema_cache or "error" in self.schema_cache:
            return "Schema not available"

        llm_format = []
        llm_format.append("DATABASE SCHEMA:\n")

        for table_name, table_info in self.schema_cache["tables"].items():
            columns = [f"{col['name']} ({col['type']})" for col in table_info["columns"]]
            llm_format.append(f"Table: {table_name}")
            llm_format.append(f"  Columns: {', '.join(columns)}")

            if table_info["foreign_keys"]:
                fks = [f"{fk['column']}→{fk['references_table']}" for fk in table_info["foreign_keys"]]
                llm_format.append(f"  Foreign Keys: {', '.join(fks)}")

            llm_format.append("")

        return "\n".join(llm_format)


# Helper function
def get_schema_discovery(db_connector: DatabaseConnector) -> SchemaDiscovery:
    """Create schema discovery instance."""
    return SchemaDiscovery(db_connector)
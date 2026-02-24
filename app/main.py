import streamlit as st
import os
from dotenv import load_dotenv
from db_connector import get_db_connector
from schema_discovery import get_schema_discovery

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title=os.getenv("APP_TITLE", "Database Copilot"),
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'schema' not in st.session_state:
    st.session_state.schema = None
if 'db_connector' not in st.session_state:
    st.session_state.db_connector = None

def connect_to_database():
    """Establish database connection."""
    with st.spinner("🔌 Connecting to database..."):
        db = get_db_connector()
        success = db.connect()

        if success:
            st.session_state.db_connected = True
            st.session_state.db_connector = db

            # Test connection
            status = db.test_connection()
            if status["connected"]:
                st.success(f"✅ Connected to: {status['database']}")
                st.info(f"📊 PostgreSQL Version: {status['version']}")
                return True
            else:
                st.error(f"❌ Connection test failed: {status.get('error', 'Unknown error')}")
                return False
        else:
            st.error("❌ Failed to establish database connection")
            return False

def discover_schema():
    """Discover database schema."""
    if not st.session_state.db_connected:
        st.warning("⚠️ Please connect to database first")
        return

    with st.spinner("🔍 Discovering database schema..."):
        schema_disco = get_schema_discovery(st.session_state.db_connector)
        schema = schema_disco.discover_full_schema()

        if "error" not in schema:
            st.session_state.schema = schema
            st.session_state.schema_discovery = schema_disco
            st.success(f"✅ Discovered {schema['total_tables']} tables with {schema['total_columns']} columns")
            return True
        else:
            st.error(f"❌ Schema discovery failed: {schema['error']}")
            return False

def main():
    # Header
    st.markdown('<p class="main-header">🤖 Database Copilot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Talk to your PostgreSQL database in plain English</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Connection status
        st.markdown("---")
        st.markdown("### 📊 Status")

        if st.session_state.db_connected:
            st.success("✅ Database: Connected")
        else:
            st.error("❌ Database: Not Connected")

        st.info("ℹ️ Ollama: Running")

        if st.session_state.schema:
            st.success(f"✅ Schema: {st.session_state.schema['total_tables']} tables")
        else:
            st.warning("⏳ Schema: Not Loaded")

        # Connection button
        st.markdown("---")
        if not st.session_state.db_connected:
            if st.button("🔌 Connect to Database", use_container_width=True):
                connect_to_database()
        else:
            if st.button("🔄 Reconnect", use_container_width=True):
                st.session_state.db_connected = False
                st.session_state.schema = None
                st.rerun()

        # Schema discovery button
        if st.session_state.db_connected and not st.session_state.schema:
            if st.button("🔍 Discover Schema", use_container_width=True):
                discover_schema()

        st.markdown("---")
        st.markdown("### 📖 Quick Guide")
        st.markdown("""
        **Setup Steps:**
        1. ✅ Connect to database
        2. ✅ Discover schema
        3. 🚀 Start chatting (coming soon!)

        **Example questions:**
        - "Show me all tables"
        - "How many customers?"
        - "What's total revenue?"
        """)

    # Main content area
    st.markdown("### 🚀 Database Connection")

    # Connection metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        status = "Connected" if st.session_state.db_connected else "Disconnected"
        delta_color = "normal" if st.session_state.db_connected else "off"
        st.metric(
            label="Database Status",
            value=status,
            delta="Ready" if st.session_state.db_connected else "Waiting",
            delta_color=delta_color
        )

    with col2:
        table_count = st.session_state.schema['total_tables'] if st.session_state.schema else 0
        st.metric(
            label="Tables Discovered",
            value=table_count,
            delta="Loaded" if st.session_state.schema else "Pending",
            delta_color="normal" if st.session_state.schema else "off"
        )

    with col3:
        st.metric(
            label="LLM Model",
            value="llama3.1:8b",
            delta="Ready",
            delta_color="normal"
        )

    st.markdown("---")

    # Show schema if available
    if st.session_state.schema:
        st.markdown("### 📊 Database Schema")

        # Schema summary
        summary = st.session_state.schema_discovery.get_schema_summary()
        st.markdown(summary)

        st.markdown("---")

        # Detailed table view
        st.markdown("### 📋 Table Details")

        table_names = list(st.session_state.schema['tables'].keys())
        selected_table = st.selectbox("Select a table to view details:", table_names)

        if selected_table:
            details = st.session_state.schema_discovery.get_table_details(selected_table)
            st.markdown(details)

            # Show sample data
            with st.expander(f"📄 View Sample Data from {selected_table}"):
                try:
                    query = f"SELECT * FROM {selected_table} LIMIT 5"
                    results = st.session_state.db_connector.execute_query(query)

                    if results:
                        import pandas as pd
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No data in this table")

                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")

    else:
        # Instructions when not connected
        st.info("👋 Welcome! Click **'Connect to Database'** in the sidebar to get started.")

        # Show environment info
        with st.expander("🧪 Environment Configuration"):
            st.code(f"""
DB_HOST: {os.getenv('DB_HOST', 'Not Set')}
DB_PORT: {os.getenv('DB_PORT', 'Not Set')}
DB_NAME: {os.getenv('DB_NAME', 'Not Set')}
DB_USER: {os.getenv('DB_USER', 'Not Set')}
OLLAMA_HOST: {os.getenv('OLLAMA_HOST', 'Not Set')}
            """)

if __name__ == "__main__":
    main()
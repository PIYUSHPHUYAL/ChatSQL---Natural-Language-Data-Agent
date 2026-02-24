-- Sample e-commerce database for testing

-- Customers table
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL
);

-- Products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    total DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL
);

-- Order items table
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL
);

-- Insert sample customers
INSERT INTO customers (name, email) VALUES
('John Doe', 'john@example.com'),
('Jane Smith', 'jane@example.com'),
('Bob Wilson', 'bob@example.com'),
('Alice Brown', 'alice@example.com'),
('Charlie Davis', 'charlie@example.com');

-- Insert sample products
INSERT INTO products (name, price, category) VALUES
('Laptop', 999.99, 'Electronics'),
('Mouse', 29.99, 'Electronics'),
('Keyboard', 79.99, 'Electronics'),
('Monitor', 299.99, 'Electronics'),
('Desk Chair', 199.99, 'Furniture'),
('Standing Desk', 499.99, 'Furniture');

-- Insert sample orders (mix of statuses)
INSERT INTO orders (customer_id, total, status, order_date) VALUES
(1, 1079.98, 'completed', CURRENT_TIMESTAMP - INTERVAL '5 days'),
(2, 379.98, 'completed', CURRENT_TIMESTAMP - INTERVAL '3 days'),
(1, 29.99, 'refunded', CURRENT_TIMESTAMP - INTERVAL '2 days'),
(3, 699.98, 'completed', CURRENT_TIMESTAMP - INTERVAL '1 day'),
(4, 199.99, 'cancelled', CURRENT_TIMESTAMP - INTERVAL '1 day'),
(2, 999.99, 'completed', CURRENT_TIMESTAMP),
(5, 579.98, 'pending', CURRENT_TIMESTAMP);

-- Insert order items
INSERT INTO order_items (order_id, product_id, quantity, price) VALUES
(1, 1, 1, 999.99),
(1, 2, 1, 29.99),
(1, 3, 1, 79.99),
(2, 4, 1, 299.99),
(2, 3, 1, 79.99),
(3, 2, 1, 29.99),
(4, 5, 1, 199.99),
(4, 6, 1, 499.99),
(5, 5, 1, 199.99),
(6, 1, 1, 999.99),
(7, 6, 1, 499.99),
(7, 3, 1, 79.99);
-- SQLite database schema for GPT-Engineer Web UI
-- File: schema.sql

-- Users table
CREATE TABLE user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(80) NOT NULL UNIQUE,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(128) NOT NULL,
    is_admin BOOLEAN NOT NULL DEFAULT 0,
    api_key VARCHAR(64) UNIQUE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    is_active BOOLEAN NOT NULL DEFAULT 1
);

-- Create index on username and email
CREATE INDEX idx_user_username ON user(username);
CREATE INDEX idx_user_email ON user(email);

-- Projects table
CREATE TABLE project (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    directory VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_public BOOLEAN NOT NULL DEFAULT 0,
    user_id INTEGER,
    prompt TEXT,
    model VARCHAR(50),
    status VARCHAR(20) DEFAULT 'idle',
    metadata JSON,
    tags TEXT,
    iteration_count INTEGER DEFAULT 0,
    rating INTEGER,
    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE SET NULL
);

-- Create indexes for projects
CREATE INDEX idx_project_user_id ON project(user_id);
CREATE INDEX idx_project_status ON project(status);
CREATE INDEX idx_project_is_public ON project(is_public);

-- Job history table
CREATE TABLE job_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    job_type VARCHAR(20) NOT NULL,
    model VARCHAR(50) NOT NULL,
    prompt TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    start_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,
    duration_seconds INTEGER,
    error TEXT,
    output_summary TEXT,
    files_changed INTEGER DEFAULT 0,
    token_usage INTEGER,
    FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
);

-- Create indexes for job history
CREATE INDEX idx_job_history_project_id ON job_history(project_id);
CREATE INDEX idx_job_history_status ON job_history(status);
CREATE INDEX idx_job_history_start_time ON job_history(start_time);

-- Insert admin user with password 'admin'
INSERT INTO user (username, email, password_hash, is_admin, api_key, is_active)
VALUES ('admin', 'admin@example.com', 'pbkdf2:sha256:150000$XXXXXXXX$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', 1, 'sk-adminxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', 1);

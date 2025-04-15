# GPT-Engineer Web UI

A comprehensive web interface for [GPT-Engineer](https://github.com/AntonOsika/gpt-engineer), allowing you to create and improve AI-generated coding projects through an intuitive UI.

![image](https://github.com/user-attachments/assets/dae64fa3-6802-443c-98c9-88c15514f2f3)


## Features

- **User Management**: Authentication, roles, and API keys
- **Project Management**: Create, improve, clone, and delete projects
- **Templates**: Start projects from pre-defined templates
- **Job Processing**: Background execution with status tracking
- **File Browser**: View and explore generated code
- **API**: RESTful endpoints for all operations


![image](https://github.com/user-attachments/assets/6424a134-557a-4d8e-a350-a52f8b1620de)
![image](https://github.com/user-attachments/assets/4b01a5f9-cf9d-4335-b652-ad15c242b701)
![image](https://github.com/user-attachments/assets/a8427be9-a052-4755-ae48-0f2ee4f5bb81)

## Installation

### Prerequisites
- Python 3.8+
- GPT-Engineer

### Option 1: Local Installation

1. Clone the repository
   ```bash
   git clone https://github.com/DeepseekAI-sketch/gpt-engineer-webui.git
   cd gpt-engineer-webui
 ```
1.1 Make change in App.py do change OPENAI_API_KEY 'sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' to your
```bash
        'OPENAI_API_BASE': os.getenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', 'sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'),
        'MODEL_NAME': os.getenv('MODEL_NAME', 'google/gemini-2.0-flash-thinking-exp:free')
 ```
        
1. Run the Application
   ```bash
   python -m venv venv
   venv\Scripts\activate
   python app.py
    ```
Or
   ```bash
   python app.py
 ```

Here's a comprehensive `README.md` for your GPT-Engineer Web UI project:


# GPT-Engineer Web UI

A production-ready web interface for GPT-Engineer, enabling users to create, manage, and improve AI-generated coding projects through an intuitive UI.

![Project Logo](static/img/logo.png)

## Features

- **User Management System**
  - Registration, login, and profile management
  - Role-based access control (admin/regular users)
  - API key generation for programmatic access

- **Project Management**
  - Create projects from templates
  - Upload reference resources
  - Browse and edit generated files
  - Improve existing projects with new prompts
  - Rate and tag projects

- **Job Processing**
  - Background execution of GPT-Engineer
  - Real-time status tracking
  - Comprehensive job history
  - Timeout and error handling

- **Administration Dashboard**
  - User management
  - System monitoring
  - Backup tools
  - Log viewing

- **API Access**
  - RESTful endpoints for all operations
  - Batch processing capabilities
  - API key authentication

## Project Structure

```
gpt-engineer-webui/
│
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── .env                      # Environment config
├── .gitignore                # Git ignore rules
├── app.log                   # Application logs
│
├── static/                   # Static assets (CSS, JS, images)
├── templates/                # Jinja2 HTML templates
├── projects/                 # Project storage
├── uploads/                  # Temporary uploads
├── temp/                     # Temporary files
├── backups/                  # Backup storage
├── templates/project_templates/ # Project templates
└── instance/                 # Database instance
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gpt-engineer-webui.git
   cd gpt-engineer-webui
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```ini
   FLASK_APP=app.py
   FLASK_ENV=development
   SECRET_KEY=your-secret-key
   OPENAI_API_KEY=your-openai-key
   DATABASE_URI=sqlite:///instance/gpte.db
   ```

5. Initialize the database:
   ```bash
   flask init-db
   ```

6. Run the application:
   ```bash
   flask run
   ```

## Usage

1. Access the web interface at `http://localhost:8080`
2. Register a new account or log in with existing credentials
3. Create a new project or browse existing ones
4. Submit prompts and manage generated code
5. Improve projects with additional prompts

## API Documentation

The API is available at `/api/v1/` with the following endpoints:

- `POST /api/v1/projects` - Create new project
- `GET /api/v1/projects` - List all projects
- `GET /api/v1/projects/<id>` - Get project details
- `POST /api/v1/projects/<id>/improve` - Improve existing project
- `GET /api/v1/users` - User management (admin only)

All API requests require an `X-API-KEY` header with a valid API key.

## Configuration

Key configuration options in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key
- `MAX_CONTENT_LENGTH`: Maximum upload size (default 16MB)
- `SESSION_TIMEOUT`: User session timeout in minutes
- `BACKUP_DIR`: Backup directory path
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Deployment

For production deployment:

1. Set `FLASK_ENV=production` in `.env`
2. Configure a production WSGI server (Gunicorn, uWSGI)
3. Set up a reverse proxy (Nginx, Apache)
4. Configure HTTPS with valid certificates

Example Gunicorn command:
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Backup and Restore

Automated backups are stored in the `backups/` directory. To manually create a backup:

```bash
flask backup
```

To restore from a backup:

```bash
flask restore backup_filename.zip
```

## Contributing

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT License - See [LICENSE](LICENSE) for details.
```

This README includes:
1. Project overview
2. Key features
3. Directory structure
4. Installation instructions
5. Usage guide
6. API documentation
7. Configuration options
8. Deployment instructions
9. Backup procedures
10. Contribution guidelines
11. License information

You may want to add additional sections like:
- Screenshots
- Troubleshooting
- Roadmap
- Acknowledgements
- Version history

The markdown is formatted for good readability on GitHub and includes clear section headers and code blocks for commands.
   


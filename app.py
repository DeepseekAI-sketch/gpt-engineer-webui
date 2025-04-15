import os
import json
import time
import uuid
import shutil
import logging
import secrets
import threading
import subprocess
from flask import current_app
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask import session, send_from_directory, abort, Response, g
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_caching import Cache
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Sentry for error tracking (if configured)
if os.getenv('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=os.getenv('SENTRY_DSN'),
        integrations=[FlaskIntegration()],
        traces_sample_rate=0.3,
        environment=os.getenv('FLASK_ENV', 'development')
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO if os.getenv('FLASK_ENV') == 'production' else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask application setup
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=os.getenv('SECRET_KEY', secrets.token_hex(32)),
    MAX_CONTENT_LENGTH=int(os.getenv('MAX_UPLOAD_SIZE', 16)) * 1024 * 1024,
    UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', 'uploads'),
    PROJECT_FOLDER=os.getenv('PROJECT_FOLDER', 'projects'),
    TEMP_FOLDER=os.getenv('TEMP_FOLDER', 'temp'),
    BACKUP_FOLDER=os.getenv('BACKUP_FOLDER', 'backups'),
    ALLOWED_EXTENSIONS=set(os.getenv('ALLOWED_EXTENSIONS', 'txt,md,py,js,css,html,json,yaml,yml').split(',')),
    SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL', 'sqlite:///gpte.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_TYPE='filesystem',
    SESSION_COOKIE_SECURE=os.getenv('FLASK_ENV') == 'production',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    CORS_ORIGINS=os.getenv('CORS_ORIGINS', '*').split(','),
    DEBUG=os.getenv('FLASK_ENV') != 'production',
    CACHE_TYPE='SimpleCache',
    CACHE_DEFAULT_TIMEOUT=300,
    GPT_MODELS=json.loads(os.getenv('GPT_MODELS', '{"gpt-4o": "GPT-4o", "gpt-4-turbo": "GPT-4 Turbo", "gpt-3.5-turbo": "GPT-3.5 Turbo"}')),
    API_CONFIG={
        'OPENAI_API_BASE': os.getenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', 'sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'),
        'MODEL_NAME': os.getenv('MODEL_NAME', 'google/gemini-2.0-flash-thinking-exp:free')
    },
    PROJECT_TEMPLATES=os.getenv('PROJECT_TEMPLATES', 'templates/project_templates'),
    ENABLE_AUTHENTICATION=os.getenv('ENABLE_AUTHENTICATION', 'false').lower() == 'true',
    MAX_CONCURRENT_JOBS=int(os.getenv('MAX_CONCURRENT_JOBS', 3)),
    JOB_TIMEOUT_MINUTES=int(os.getenv('JOB_TIMEOUT_MINUTES', 30)),
    ENABLE_ANALYTICS=os.getenv('ENABLE_ANALYTICS', 'false').lower() == 'true',
    AUTO_BACKUP_ENABLED=os.getenv('AUTO_BACKUP_ENABLED', 'true').lower() == 'true',
    AUTO_BACKUP_INTERVAL_HOURS=int(os.getenv('AUTO_BACKUP_INTERVAL_HOURS', 24)),
    RATE_LIMITING_ENABLED=os.getenv('RATE_LIMITING_ENABLED', 'true').lower() == 'true'
)

# Initialize extensions
db = SQLAlchemy(app)
cache = Cache(app)
cors = CORS(app)
scheduler = BackgroundScheduler()

# Rate limiting
if app.config['RATE_LIMITING_ENABLED']:
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )

# Set up login manager if authentication is enabled
if app.config['ENABLE_AUTHENTICATION']:
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'

# Ensure directories exist
for directory in [app.config['UPLOAD_FOLDER'], app.config['PROJECT_FOLDER'], 
                 app.config['TEMP_FOLDER'], app.config['BACKUP_FOLDER']]:
    os.makedirs(directory, exist_ok=True)

# Database models
if app.config['ENABLE_AUTHENTICATION']:
    class User(UserMixin, db.Model):
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False)
        email = db.Column(db.String(120), unique=True, nullable=False)
        password_hash = db.Column(db.String(128))
        is_admin = db.Column(db.Boolean, default=False)
        api_key = db.Column(db.String(64), unique=True)
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        last_login = db.Column(db.DateTime)
        is_active = db.Column(db.Boolean, default=True)
        projects = db.relationship('Project', backref='owner', lazy=True)

        def set_password(self, password):
            self.password_hash = generate_password_hash(password)

        def check_password(self, password):
            return check_password_hash(self.password_hash, password)

        def generate_api_key(self):
            self.api_key = secrets.token_hex(32)
            return self.api_key

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    directory = db.Column(db.String(255), unique=True, nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_modified = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public = db.Column(db.Boolean, default=False)
    
    if app.config['ENABLE_AUTHENTICATION']:
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Latest prompt
    prompt = db.Column(db.Text)
    # Model used for latest generation
    model = db.Column(db.String(50))
    # Latest job status
    status = db.Column(db.String(20), default='idle')
    # Custom metadata (JSON) - renamed to project_metadata to avoid conflict
    project_metadata = db.Column(db.JSON)  # Changed from 'metadata' to 'project_metadata'
    # Tags for filtering
    tags = db.Column(db.Text)
    # Number of improvement iterations
    iteration_count = db.Column(db.Integer, default=0)
    # Star rating (1-5)
    rating = db.Column(db.Integer)
    
    def to_dict(self):
        """Convert project to dictionary for API responses"""
        result = {
            'id': self.id,
            'name': self.name,
            'directory': self.directory,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            'is_public': self.is_public,
            'status': self.status,
            'prompt': self.prompt,
            'model': self.model,
            'tags': self.tags.split(',') if self.tags else [],
            'iteration_count': self.iteration_count,
            'rating': self.rating,
            'metadata': self.project_metadata  # Keep the API response using 'metadata' for backward compatibility
        }
        
        if app.config['ENABLE_AUTHENTICATION']:
            result['user_id'] = self.user_id
            
            return result
        
class JobHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    job_type = db.Column(db.String(20), nullable=False)  # 'create' or 'improve'
    model = db.Column(db.String(50), nullable=False)
    prompt = db.Column(db.Text)
    status = db.Column(db.String(20), default='pending')
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    duration_seconds = db.Column(db.Integer)
    error = db.Column(db.Text)
    output_summary = db.Column(db.Text)
    files_changed = db.Column(db.Integer, default=0)
    token_usage = db.Column(db.Integer)
    
    project = db.relationship('Project', backref='jobs')
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'job_type': self.job_type,
            'model': self.model,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'error': self.error,
            'files_changed': self.files_changed,
            'token_usage': self.token_usage
        }

# Create the database tables
with app.app_context():
    db.create_all()

# Track running jobs
active_jobs = {}

# User login/authentication functions
if app.config['ENABLE_AUTHENTICATION']:
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            remember = 'remember' in request.form
            
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password):
                login_user(user, remember=remember)
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
            else:
                flash('Invalid username or password', 'danger')
                
        return render_template('login.html')
        
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('index'))
        
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if password != confirm_password:
                flash('Passwords do not match', 'danger')
                return redirect(url_for('register'))
                
            if User.query.filter_by(username=username).first():
                flash('Username already exists', 'danger')
                return redirect(url_for('register'))
                
            if User.query.filter_by(email=email).first():
                flash('Email already registered', 'danger')
                return redirect(url_for('register'))
                
            user = User(username=username, email=email)
            user.set_password(password)
            user.generate_api_key()
            
            # First user becomes admin
            if User.query.count() == 0:
                user.is_admin = True
                
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful. You can now log in.', 'success')
            return redirect(url_for('login'))
            
        return render_template('register.html')
        
    @app.route('/profile')
    @login_required
    def profile():
        # User's projects
        projects = Project.query.filter_by(user_id=current_user.id).order_by(Project.last_modified.desc()).all()
        # User's job history
        job_history = JobHistory.query.join(Project).filter(Project.user_id == current_user.id).order_by(JobHistory.start_time.desc()).limit(10).all()
        
        return render_template('profile.html', user=current_user, projects=projects, job_history=job_history)
        
    @app.route('/profile/api_key', methods=['POST'])
    @login_required
    def regenerate_api_key():
        current_user.generate_api_key()
        db.session.commit()
        flash('API key regenerated successfully', 'success')
        return redirect(url_for('profile'))
    
    # Admin panel
    @app.route('/admin')
    @login_required
    def admin_panel():
        if not current_user.is_admin:
            abort(403)
            
        users = User.query.all()
        projects = Project.query.all()
        jobs = JobHistory.query.order_by(JobHistory.start_time.desc()).limit(50).all()
        
        # System stats
        stats = {
            'user_count': User.query.count(),
            'project_count': Project.query.count(),
            'job_count': JobHistory.query.count(),
            'active_jobs': len(active_jobs),
            'disk_usage': get_disk_usage(),
            'system_load': get_system_load()
        }
        
        return render_template('admin.html', users=users, projects=projects, jobs=jobs, stats=stats)
    
    # Admin user management
    @app.route('/admin/users/<int:user_id>', methods=['GET', 'POST'])
    @login_required
    def admin_edit_user(user_id):
        if not current_user.is_admin:
            abort(403)
            
        user = User.query.get_or_404(user_id)
        
        if request.method == 'POST':
            user.username = request.form.get('username')
            user.email = request.form.get('email')
            user.is_admin = 'is_admin' in request.form
            user.is_active = 'is_active' in request.form
            
            password = request.form.get('password')
            if password:
                user.set_password(password)
                
            db.session.commit()
            flash(f'User {user.username} updated successfully', 'success')
            return redirect(url_for('admin_panel'))
            
        return render_template('admin_edit_user.html', user=user)

# API Key authentication for API routes
def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key is required'}), 401
            
        user = User.query.filter_by(api_key=api_key).first()
        if not user or not user.is_active:
            return jsonify({'error': 'Invalid or inactive API key'}), 401
            
        g.current_user = user
        return f(*args, **kwargs)
    return decorated_function

# Utility functions
def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_disk_usage():
    """Get disk usage statistics for the app directories"""
    total_size = 0
    for directory in [app.config['PROJECT_FOLDER'], app.config['UPLOAD_FOLDER'], app.config['TEMP_FOLDER']]:
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    
    # Convert to MB
    return round(total_size / (1024 * 1024), 2)

def get_system_load():
    """Get system load information"""
    try:
        import psutil
        return {
            'cpu': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent
        }
    except ImportError:
        return {'cpu': 'N/A', 'memory': 'N/A', 'disk': 'N/A'}

@app.context_processor
def inject_current_app():
    return dict(app=current_app)


def run_gpt_engineer(project_dir, model="gpt-4o", task_type="create", callback=None, metadata=None, user_id=None):
    """
    Run GPT-Engineer process
    
    Args:
        project_dir: The directory of the project
        model: The model to use
        task_type: "create" or "improve"
        callback: Function to call when process completes
        metadata: Additional metadata for the job
        user_id: ID of the user who initiated the job
    """
    try:
        # Check if we've reached max concurrent jobs
        active_job_count = sum(1 for job in active_jobs.values() if job['status'] == 'running')
        if active_job_count >= app.config['MAX_CONCURRENT_JOBS']:
            logger.warning(f"Maximum concurrent jobs limit reached ({app.config['MAX_CONCURRENT_JOBS']})")
            raise Exception(f"Maximum concurrent jobs limit reached ({app.config['MAX_CONCURRENT_JOBS']}). Please try again later.")
            
        # Use project ID as job ID
        job_id = os.path.basename(project_dir)
        
        # Set environment variables for API configuration
        env = os.environ.copy()
        env.update({
            'OPENAI_API_BASE': app.config['API_CONFIG']['OPENAI_API_BASE'],
            'OPENAI_API_KEY': app.config['API_CONFIG']['OPENAI_API_KEY'],
            'MODEL_NAME': app.config['API_CONFIG']['MODEL_NAME']
        })
        
        # Additional arguments based on model and metadata
        cmd = ["gpte", project_dir, "--model", model]
        
        if task_type == "improve":
            cmd.append("-i")
        
        # Add any custom arguments based on metadata
        if metadata:
            if metadata.get('temperature'):
                cmd.extend(["--temperature", str(metadata['temperature'])])
            if metadata.get('context_window'):
                cmd.extend(["--context-window", str(metadata['context_window'])])
            if metadata.get('steps'):
                cmd.extend(["--steps", metadata['steps']])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Create a new process and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        
        # Record in database
        project = Project.query.filter_by(directory=project_dir).first()
        if project:
            job = JobHistory(
                project_id=project.id,
                job_type=task_type,
                model=model,
                prompt=project.prompt,
                status='running',
                start_time=datetime.utcnow()
            )
            db.session.add(job)
            db.session.commit()
            job_db_id = job.id
        else:
            job_db_id = None
        
        start_time = datetime.utcnow()
        timeout_time = start_time + timedelta(minutes=app.config['JOB_TIMEOUT_MINUTES'])
        
        active_jobs[job_id] = {
            'process': process,
            'output': [],
            'status': 'running',
            'start_time': start_time,
            'timeout_time': timeout_time,
            'project_dir': project_dir,
            'model': model,
            'task_type': task_type,
            'job_db_id': job_db_id,
            'metadata': metadata or {},
            'user_id': user_id
        }
        
        # Read output line by line
        for line in process.stdout:
            active_jobs[job_id]['output'].append(line)
            logger.debug(line.strip())
        
        # Wait for the process to complete
        returncode = process.wait()
        end_time = datetime.utcnow()
        
        # Update job status
        if returncode == 0:
            active_jobs[job_id]['status'] = 'completed'
            status = 'completed'
            logger.info(f"Job {job_id} completed successfully")
        else:
            active_jobs[job_id]['status'] = 'failed'
            status = 'failed'
            logger.error(f"Job {job_id} failed with return code {returncode}")
            
            # Get error output
            error = process.stderr.read()
            active_jobs[job_id]['error'] = error
            logger.error(f"Error: {error}")
        
        # Update database record
        if job_db_id:
            job = JobHistory.query.get(job_db_id)
            if job:
                job.status = status
                job.end_time = end_time
                job.duration_seconds = (end_time - job.start_time).total_seconds()
                
                if status == 'failed' and 'error' in active_jobs[job_id]:
                    job.error = active_jobs[job_id]['error']
                
                # Count files changed
                if status == 'completed':
                    workspace_dir = os.path.join(project_dir, 'workspace')
                    file_count = sum(1 for _ in Path(workspace_dir).glob('**/*') if _.is_file())
                    job.files_changed = file_count
                
                db.session.commit()
            
            # Update project status
            if project:
                project.status = status
                project.last_modified = end_time
                if task_type == 'improve':
                    project.iteration_count += 1
                db.session.commit()
        
        # Call callback if provided
        if callback:
            callback(job_id)
        
        # After some time, clean up the job from memory
        def cleanup_job():
            if job_id in active_jobs:
                # Keep basic info but remove large data
                active_jobs[job_id] = {
                    'status': active_jobs[job_id]['status'],
                    'start_time': active_jobs[job_id]['start_time'],
                    'end_time': end_time,
                    'model': model,
                    'task_type': task_type,
                    'job_db_id': job_db_id,
                    'cleaned_up': True
                }
        
        # Schedule cleanup after 30 minutes
        timer = threading.Timer(1800, cleanup_job)
        timer.daemon = True
        timer.start()
            
    except Exception as e:
        logger.exception(f"Error running GPT-Engineer: {str(e)}")
        if 'job_id' in locals():  # Fix: Check if job_id is defined before using it
            if job_id in active_jobs:
                active_jobs[job_id]['status'] = 'failed'
                active_jobs[job_id]['error'] = str(e)
            
        # Update database record
        if 'job_db_id' in locals() and job_db_id:
            job = JobHistory.query.get(job_db_id)
            if job:
                job.status = 'failed'
                job.end_time = datetime.utcnow()
                job.duration_seconds = (job.end_time - job.start_time).total_seconds()
                job.error = str(e)
                db.session.commit()
                
            if 'project' in locals() and project:
                project.status = 'failed'
                db.session.commit()

def backup_projects():
    """Create a backup of all projects"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(app.config['BACKUP_FOLDER'], f'backup_{timestamp}.zip')
        
        # Create a zip file with all projects
        shutil.make_archive(
            backup_path.replace('.zip', ''),
            'zip',
            app.config['PROJECT_FOLDER']
        )
        
        logger.info(f"Created backup: {backup_path}")
        
        # Clean up old backups (keep last 10)
        backups = sorted(Path(app.config['BACKUP_FOLDER']).glob('backup_*.zip'))
        if len(backups) > 10:
            for old_backup in backups[:-10]:
                os.remove(old_backup)
                logger.info(f"Removed old backup: {old_backup}")
                
    except Exception as e:
        logger.exception(f"Error creating backup: {str(e)}")

def check_jobs_timeout():
    """Check for jobs that have timed out"""
    now = datetime.utcnow()
    for job_id, job in list(active_jobs.items()):
        if job['status'] == 'running' and 'timeout_time' in job and now > job['timeout_time']:
            logger.warning(f"Job {job_id} timed out after {app.config['JOB_TIMEOUT_MINUTES']} minutes")
            
            # Kill the process
            if 'process' in job:
                try:
                    job['process'].kill()
                except:
                    pass
                    
            job['status'] = 'failed'
            job['error'] = f"Job timed out after {app.config['JOB_TIMEOUT_MINUTES']} minutes"
            
            # Update database
            if 'job_db_id' in job and job['job_db_id']:
                db_job = JobHistory.query.get(job['job_db_id'])
                if db_job:
                    db_job.status = 'failed'
                    db_job.end_time = now
                    db_job.duration_seconds = (now - db_job.start_time).total_seconds()
                    db_job.error = f"Job timed out after {app.config['JOB_TIMEOUT_MINUTES']} minutes"
                    db.session.commit()
                    
                    # Update project status
                    project = Project.query.get(db_job.project_id)
                    if project:
                        project.status = 'failed'
                        db.session.commit()

# Initialize scheduled tasks
def init_scheduler():
    if app.config['AUTO_BACKUP_ENABLED']:
        # Schedule automatic backups
        scheduler.add_job(
            backup_projects, 
            'interval', 
            hours=app.config['AUTO_BACKUP_INTERVAL_HOURS'],
            id='backup_job'
        )
    
    # Check for timed out jobs every minute
    scheduler.add_job(
        check_jobs_timeout,
        'interval',
        minutes=1,
        id='timeout_check'
    )
    
    # Clean up temporary files daily
    def cleanup_temp_files():
        temp_dir = Path(app.config['TEMP_FOLDER'])
        for item in temp_dir.glob('*'):
            if item.is_file() and (datetime.now() - datetime.fromtimestamp(item.stat().st_mtime)).days > 1:
                item.unlink()
    
    scheduler.add_job(
        cleanup_temp_files,
        'interval',
        hours=24,
        id='temp_cleanup'
    )
    
    scheduler.start()

# Web Routes
@app.route('/')
def index():
    """Home page"""
    # List all projects in the project folder
    projects = []
    
    if app.config['ENABLE_AUTHENTICATION']:
        if current_user.is_authenticated:
            query = Project.query.filter_by(user_id=current_user.id)
        else:
            query = Project.query.filter_by(is_public=True)
    else:
        query = Project.query
        
    projects = query.order_by(Project.last_modified.desc()).all()
    
    # Add extra info from file system if needed
    for project in projects:
        project_dir = os.path.join(app.config['PROJECT_FOLDER'], project.directory)
        
        # Check if this project has an active job
        if project.directory in active_jobs:
            project.status = active_jobs[project.directory]['status']
    
    # Stats for dashboard
    stats = {
        'total_projects': Project.query.count(),
        'completed_jobs': JobHistory.query.filter_by(status='completed').count(),
        'active_jobs': sum(1 for job in active_jobs.values() if job['status'] == 'running')
    }
    
    return render_template('index.html', projects=projects, stats=stats)


@app.route('/create', methods=['GET', 'POST'])
def create_project():
    """Create a new project"""
    if app.config['ENABLE_AUTHENTICATION'] and not current_user.is_authenticated:
        return redirect(url_for('login', next=request.url))
        
    if request.method == 'POST':
        project_name = request.form.get('project_name', '').strip()
        project_description = request.form.get('description', '').strip()
        prompt_text = request.form.get('prompt', '').strip()
        model = request.form.get('model', 'gpt-4o')
        tags = request.form.get('tags', '').strip()
        is_public = 'is_public' in request.form
        
        # Advanced options
        temperature = request.form.get('temperature', 0.7)
        context_window = request.form.get('context_window', '')
        steps = request.form.get('steps', '')
        
        if not project_name:
            flash('Project name is required', 'danger')
            return redirect(url_for('create_project'))
            
        if not prompt_text:
            flash('Prompt is required', 'danger')
            return redirect(url_for('create_project'))
        
        # Create sanitized project directory name
        safe_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in project_name)
        # Add timestamp to ensure uniqueness
        directory_name = f"{safe_name}_{int(time.time())}"
        project_dir = os.path.join(app.config['PROJECT_FOLDER'], directory_name)
        
        # Check if project already exists
        if os.path.exists(project_dir):
            flash(f'Project "{project_name}" already exists. Please choose a different name.', 'danger')
            return redirect(url_for('create_project'))
        
        try:
            # Create project directory
            os.makedirs(project_dir, exist_ok=True)
            
            # Write prompt file
            with open(os.path.join(project_dir, 'prompt'), 'w') as f:
                f.write(prompt_text)
            
            # Create project in database
            project = Project(
                name=project_name,
                directory=directory_name,
                description=project_description,
                prompt=prompt_text,
                model=model,
                status='pending',
                tags=tags,
                is_public=is_public
            )
            
            if app.config['ENABLE_AUTHENTICATION'] and current_user.is_authenticated:
                project.user_id = current_user.id
                
            db.session.add(project)
            db.session.commit()
            
            # Handle uploaded files
            uploaded_files = request.files.getlist('files')
            if uploaded_files and any(file.filename for file in uploaded_files):
                resource_dir = os.path.join(project_dir, 'resources')
                os.makedirs(resource_dir, exist_ok=True)
                
                for file in uploaded_files:
                    if file and file.filename:
                        if allowed_file(file.filename):
                            filename = secure_filename(file.filename)
                            file.save(os.path.join(resource_dir, filename))
                        else:
                            flash(f'Ignoring file with disallowed extension: {file.filename}', 'warning')
            
            # Handle template selection
            template = request.form.get('template')
            if template and template != 'none':
                template_dir = os.path.join(app.config['PROJECT_TEMPLATES'], template)
                if os.path.exists(template_dir):
                    # Copy template files to resources directory
                    target_dir = os.path.join(project_dir, 'resources')
                    os.makedirs(target_dir, exist_ok=True)
                    for item in os.listdir(template_dir):
                        src = os.path.join(template_dir, item)
                        dst = os.path.join(target_dir, item)
                        if os.path.isfile(src):
                            shutil.copy2(src, dst)
                        elif os.path.isdir(src):
                            shutil.copytree(src, dst)
                            
                    # Add template info to metadata
                    project.project_metadata = project.project_metadata or {}
                    project.project_metadata['template'] = template
                    db.session.commit()
            
            # Gather metadata for job
            metadata = {
                'temperature': float(temperature) if temperature else 0.7,
            }
            
            if context_window:
                metadata['context_window'] = int(context_window)
                
            if steps:
                metadata['steps'] = steps
            
            # Start GPT-Engineer process in a separate thread
            user_id = current_user.id if app.config['ENABLE_AUTHENTICATION'] and current_user.is_authenticated else None
            thread = threading.Thread(
                target=run_gpt_engineer,
                args=(project_dir, model, "create"),
                kwargs={
                    'callback': lambda job_id: logger.info(f"Job {job_id} callback triggered"),
                    'metadata': metadata,
                    'user_id': user_id
                }
            )
            thread.daemon = True
            thread.start()
            
            flash(f'Project "{project_name}" created successfully. GPT-Engineer is now processing your request.', 'success')
            return redirect(url_for('project_detail', project_id=project.id))
            
        except Exception as e:
            logger.exception(f"Error creating project: {str(e)}")
            flash(f'Error creating project: {str(e)}', 'danger')
            return redirect(url_for('create_project'))
    
    # GET request
    models = []
    for model_id, model_name in app.config['GPT_MODELS'].items():
        models.append({
            'id': model_id,
            'name': model_name,
            'description': get_model_description(model_id)
        })
    
    # Get available templates
    templates = [{'id': 'none', 'name': 'No Template', 'description': 'Start from scratch'}]
    template_dir = app.config['PROJECT_TEMPLATES']
    if os.path.exists(template_dir):
        for item in os.listdir(template_dir):
            if os.path.isdir(os.path.join(template_dir, item)):
                # Read template info if available
                info_file = os.path.join(template_dir, item, 'template_info.json')
                if os.path.exists(info_file):
                    try:
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                            templates.append({
                                'id': item,
                                'name': info.get('name', item),
                                'description': info.get('description', ''),
                                'tags': info.get('tags', [])
                            })
                    except:
                        templates.append({
                            'id': item,
                            'name': item.replace('_', ' ').title(),
                            'description': ''
                        })
                else:
                    templates.append({
                        'id': item,
                        'name': item.replace('_', ' ').title(),
                        'description': ''
                    })
    
    return render_template(
        'create_project.html', 
        models=models, 
        templates=templates,
        advanced_options=True
    )


@app.route('/project/<int:project_id>')
def project_detail(project_id):
    """Show project details and status"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if not current_user.is_authenticated and not project.is_public:
            flash('You need to be logged in to view this project', 'danger')
            return redirect(url_for('login', next=request.url))
        elif current_user.is_authenticated and not current_user.is_admin and project.user_id != current_user.id and not project.is_public:
            flash('You do not have permission to view this project', 'danger')
            return redirect(url_for('index'))
    
    project_dir = os.path.join(app.config['PROJECT_FOLDER'], project.directory)
    
    if not os.path.exists(project_dir):
        flash(f'Project directory not found. The project may have been deleted from disk.', 'danger')
        return redirect(url_for('index'))
    
    # Get project files
    files = []
    workspace_dir = os.path.join(project_dir, 'workspace')
    if os.path.exists(workspace_dir):
        for root, _, filenames in os.walk(workspace_dir):
            for file in filenames:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, workspace_dir)
                
                # Get file info
                file_info = {
                    'name': rel_path,
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Only try to read content for small text files
                if os.path.getsize(file_path) < 1024 * 1024:  # 1MB limit
                    try:
                        with open(file_path, 'r') as f:
                            file_info['content'] = f.read()
                            file_info['is_binary'] = False
                    except UnicodeDecodeError:
                        file_info['content'] = 'Binary file'
                        file_info['is_binary'] = True
                else:
                    file_info['content'] = 'File too large to display'
                    file_info['is_binary'] = True
                
                files.append(file_info)
    
    # Get job history
    job_history = JobHistory.query.filter_by(project_id=project_id).order_by(JobHistory.start_time.desc()).all()
    
    # Check if this project has an active job
    output = []
    error = None
    if project.directory in active_jobs:
        active_job = active_jobs[project.directory]
        project.status = active_job['status']
        
        if 'cleaned_up' not in active_job:
            output = active_job.get('output', [])
        
        if 'error' in active_job:
            error = active_job['error']
    
    # Get resource files
    resources = []
    resource_dir = os.path.join(project_dir, 'resources')
    if os.path.exists(resource_dir):
        for root, _, filenames in os.walk(resource_dir):
            for file in filenames:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, resource_dir)
                resources.append({
                    'name': rel_path,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    # Get project dependencies if any
    dependencies = []
    requirements_file = os.path.join(workspace_dir, 'requirements.txt')
    if os.path.exists(requirements_file):
        try:
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dependencies.append(line)
        except:
            pass
    
    # Parse tags
    tags = project.tags.split(',') if project.tags else []
    
    return render_template(
        'project_detail.html', 
        project=project,
        files=files,
        resources=resources, 
        output=output,
        error=error,
        job_history=job_history,
        dependencies=dependencies,
        tags=tags,
        is_owner=app.config['ENABLE_AUTHENTICATION'] and current_user.is_authenticated and project.user_id == current_user.id,
        is_admin=app.config['ENABLE_AUTHENTICATION'] and current_user.is_authenticated and current_user.is_admin
    )


@app.route('/project/<int:project_id>/improve', methods=['GET', 'POST'])
def improve_project(project_id):
    """Improve an existing project"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if not current_user.is_authenticated:
            flash('You need to be logged in to improve projects', 'danger')
            return redirect(url_for('login', next=request.url))
        elif not current_user.is_admin and project.user_id != current_user.id:
            flash('You do not have permission to improve this project', 'danger')
            return redirect(url_for('index'))
    
    project_dir = os.path.join(app.config['PROJECT_FOLDER'], project.directory)
    
    if not os.path.exists(project_dir):
        flash(f'Project directory not found. The project may have been deleted from disk.', 'danger')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        prompt_text = request.form.get('prompt', '').strip()
        model = request.form.get('model', 'gpt-4o')
        temperature = request.form.get('temperature', 0.7)
        context_window = request.form.get('context_window', '')
        steps = request.form.get('steps', '')
        
        if not prompt_text:
            flash('Improvement prompt is required', 'danger')
            return redirect(url_for('improve_project', project_id=project_id))
        
        try:
            # Write prompt file
            with open(os.path.join(project_dir, 'prompt'), 'w') as f:
                f.write(prompt_text)
            
            # Update project in database
            project.prompt = prompt_text
            project.model = model
            project.status = 'pending'
            project.last_modified = datetime.utcnow()
            db.session.commit()
            
            # Gather metadata for job
            metadata = {
                'temperature': float(temperature) if temperature else 0.7,
            }
            
            if context_window:
                metadata['context_window'] = int(context_window)
                
            if steps:
                metadata['steps'] = steps
            
            # Start GPT-Engineer process in a separate thread
            user_id = current_user.id if app.config['ENABLE_AUTHENTICATION'] and current_user.is_authenticated else None
            thread = threading.Thread(
                target=run_gpt_engineer,
                args=(project_dir, model, "improve"),
                kwargs={
                    'callback': lambda job_id: logger.info(f"Job {job_id} callback triggered"),
                    'metadata': metadata,
                    'user_id': user_id
                }
            )
            thread.daemon = True
            thread.start()
            
            flash(f'Project improvement started. GPT-Engineer is now processing your request.', 'success')
            return redirect(url_for('project_detail', project_id=project_id))
            
        except Exception as e:
            logger.exception(f"Error improving project: {str(e)}")
            flash(f'Error improving project: {str(e)}', 'danger')
            return redirect(url_for('improve_project', project_id=project_id))
    
    # GET request
    # Get models
    models = []
    for model_id, model_name in app.config['GPT_MODELS'].items():
        models.append({
            'id': model_id,
            'name': model_name,
            'description': get_model_description(model_id)
        })
    
    return render_template(
        'improve_project.html', 
        project=project,
        models=models,
        advanced_options=True
    )


@app.route('/project/<int:project_id>/download')
def download_project(project_id):
    """Download project as a zip file"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if not current_user.is_authenticated and not project.is_public:
            flash('You need to be logged in to download this project', 'danger')
            return redirect(url_for('login', next=request.url))
        elif current_user.is_authenticated and not current_user.is_admin and project.user_id != current_user.id and not project.is_public:
            flash('You do not have permission to download this project', 'danger')
            return redirect(url_for('index'))
    
    project_dir = os.path.join(app.config['PROJECT_FOLDER'], project.directory)
    
    if not os.path.exists(project_dir):
        flash(f'Project directory not found. The project may have been deleted from disk.', 'danger')
        return redirect(url_for('index'))
    
    # Create a temporary zip file
    temp_dir = app.config['TEMP_FOLDER']
    os.makedirs(temp_dir, exist_ok=True)
    
    zip_filename = f"{project.name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.zip"
    zip_path = os.path.join(temp_dir, zip_filename)
    
    try:
        # Create zip file
        shutil.make_archive(zip_path[:-4], 'zip', project_dir)
        
        # Return the file for download
        return send_from_directory(temp_dir, zip_filename, as_attachment=True)
    except Exception as e:
        logger.exception(f"Error creating zip file: {str(e)}")
        flash(f'Error creating zip file: {str(e)}', 'danger')
        return redirect(url_for('project_detail', project_id=project_id))


@app.route('/project/<int:project_id>/clone', methods=['POST'])
def clone_project(project_id):
    """Clone an existing project"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if not current_user.is_authenticated:
            flash('You need to be logged in to clone projects', 'danger')
            return redirect(url_for('login'))
        elif not current_user.is_admin and not project.is_public and project.user_id != current_user.id:
            flash('You do not have permission to clone this project', 'danger')
            return redirect(url_for('index'))
    
    project_dir = os.path.join(app.config['PROJECT_FOLDER'], project.directory)
    
    if not os.path.exists(project_dir):
        flash(f'Project directory not found. The project may have been deleted from disk.', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Create a new sanitized directory name
        new_name = f"{project.name} (Clone)"
        safe_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in new_name)
        # Add timestamp to ensure uniqueness
        directory_name = f"{safe_name}_{int(time.time())}"
        new_project_dir = os.path.join(app.config['PROJECT_FOLDER'], directory_name)
        
        # Copy the project directory
        shutil.copytree(project_dir, new_project_dir)
        
        # Create new project in database
        new_project = Project(
            name=new_name,
            directory=directory_name,
            description=project.description,
            prompt=project.prompt,
            model=project.model,
            status='idle',
            tags=project.tags,
            is_public=False,  # Default to private for cloned projects
        project_metadata=project.project_metadata
        )
        
        if app.config['ENABLE_AUTHENTICATION'] and current_user.is_authenticated:
            new_project.user_id = current_user.id
            
        db.session.add(new_project)
        db.session.commit()
        
        flash(f'Project "{project.name}" cloned successfully as "{new_name}"', 'success')
        return redirect(url_for('project_detail', project_id=new_project.id))
        
    except Exception as e:
        logger.exception(f"Error cloning project: {str(e)}")
        flash(f'Error cloning project: {str(e)}', 'danger')
        return redirect(url_for('project_detail', project_id=project_id))


@app.route('/project/<int:project_id>/delete', methods=['POST'])
def delete_project(project_id):
    """Delete a project"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if not current_user.is_authenticated:
            flash('You need to be logged in to delete projects', 'danger')
            return redirect(url_for('login'))
        elif not current_user.is_admin and project.user_id != current_user.id:
            flash('You do not have permission to delete this project', 'danger')
            return redirect(url_for('index'))
    
    project_dir = os.path.join(app.config['PROJECT_FOLDER'], project.directory)
    
    try:
        # Check if project has an active job
        if project.directory in active_jobs and active_jobs[project.directory]['status'] == 'running':
            # Kill the process
            try:
                active_jobs[project.directory]['process'].kill()
                active_jobs[project.directory]['status'] = 'terminated'
            except:
                pass
        
        # Delete the project directory if it exists
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
        
        # Delete job history
        JobHistory.query.filter_by(project_id=project_id).delete()
        
        # Delete the project from database
        db.session.delete(project)
        db.session.commit()
        
        flash(f'Project "{project.name}" deleted successfully', 'success')
    except Exception as e:
        logger.exception(f"Error deleting project: {str(e)}")
        flash(f'Error deleting project: {str(e)}', 'danger')
    
    return redirect(url_for('index'))


@app.route('/project/<int:project_id>/rate', methods=['POST'])
def rate_project(project_id):
    """Rate a project"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        elif not current_user.is_admin and project.user_id != current_user.id:
            return jsonify({'error': 'Permission denied'}), 403
    
    try:
        rating = int(request.form.get('rating'))
        if rating < 1 or rating > 5:
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
            
        project.rating = rating
        db.session.commit()
        
        return jsonify({'success': True, 'rating': rating})
    except Exception as e:
        logger.exception(f"Error rating project: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/project/<int:project_id>/edit', methods=['POST'])
def edit_project(project_id):
    """Edit project details"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if not current_user.is_authenticated:
            flash('You need to be logged in to edit projects', 'danger')
            return redirect(url_for('login'))
        elif not current_user.is_admin and project.user_id != current_user.id:
            flash('You do not have permission to edit this project', 'danger')
            return redirect(url_for('index'))
    
    try:
        project.name = request.form.get('name', project.name).strip()
        project.description = request.form.get('description', '').strip()
        project.tags = request.form.get('tags', '').strip()
        project.is_public = 'is_public' in request.form
        
        db.session.commit()
        
        flash(f'Project "{project.name}" updated successfully', 'success')
        return redirect(url_for('project_detail', project_id=project_id))
    except Exception as e:
        logger.exception(f"Error editing project: {str(e)}")
        flash(f'Error editing project: {str(e)}', 'danger')
        return redirect(url_for('project_detail', project_id=project_id))


# API Routes
@app.route('/api/projects', methods=['GET'])
def api_projects():
    """API: Get all projects"""
    if app.config['ENABLE_AUTHENTICATION']:
        if request.headers.get('X-API-Key'):
            return api_key_required(lambda: api_projects_impl())()
        elif not current_user.is_authenticated:
            query = Project.query.filter_by(is_public=True)
        else:
            query = Project.query.filter((Project.user_id == current_user.id) | (Project.is_public == True))
    else:
        query = Project.query
        
    projects = query.order_by(Project.last_modified.desc()).all()
    return jsonify([p.to_dict() for p in projects])


def api_projects_impl():
    """Implementation for API projects endpoint with API key auth"""
    if app.config['ENABLE_AUTHENTICATION']:
        query = Project.query.filter((Project.user_id == g.current_user.id) | (Project.is_public == True))
    else:
        query = Project.query
        
    projects = query.order_by(Project.last_modified.desc()).all()
    return jsonify([p.to_dict() for p in projects])


@app.route('/api/projects', methods=['POST'])
def api_create_project():
    """API: Create a new project"""
    if app.config['ENABLE_AUTHENTICATION']:
        if request.headers.get('X-API-Key'):
            return api_key_required(lambda: api_create_project_impl())()
        elif not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
    
    return api_create_project_impl()


def api_create_project_impl():
    """Implementation for API create project endpoint"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    if 'name' not in data or 'prompt' not in data:
        return jsonify({'error': 'Name and prompt are required'}), 400
        
    try:
        # Create sanitized project directory name
        safe_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in data['name'])
        # Add timestamp to ensure uniqueness
        directory_name = f"{safe_name}_{int(time.time())}"
        project_dir = os.path.join(app.config['PROJECT_FOLDER'], directory_name)
        
        # Create project directory
        os.makedirs(project_dir, exist_ok=True)
        
        # Write prompt file
        with open(os.path.join(project_dir, 'prompt'), 'w') as f:
            f.write(data['prompt'])
        
        # Create project in database
        project = Project(
            name=data['name'],
            directory=directory_name,
            description=data.get('description', ''),
            prompt=data['prompt'],
            model=data.get('model', 'gpt-4o'),
            status='pending',
            tags=data.get('tags', ''),
            is_public=data.get('is_public', False)
        )
        
        if app.config['ENABLE_AUTHENTICATION']:
            if hasattr(g, 'current_user'):
                project.user_id = g.current_user.id
            elif current_user.is_authenticated:
                project.user_id = current_user.id
        
        db.session.add(project)
        db.session.commit()
        
        # Gather metadata for job
        metadata = {
            'temperature': float(data.get('temperature', 0.7)),
        }
        
        if 'context_window' in data:
            metadata['context_window'] = int(data['context_window'])
            
        if 'steps' in data:
            metadata['steps'] = data['steps']
        
        # Start GPT-Engineer process in a separate thread
        user_id = None
        if app.config['ENABLE_AUTHENTICATION']:
            if hasattr(g, 'current_user'):
                user_id = g.current_user.id
            elif current_user.is_authenticated:
                user_id = current_user.id
                
        thread = threading.Thread(
            target=run_gpt_engineer,
            args=(project_dir, data.get('model', 'gpt-4o'), "create"),
            kwargs={
                'metadata': metadata,
                'user_id': user_id
            }
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Project "{data["name"]}" created successfully',
            'project': project.to_dict()
        })
        
    except Exception as e:
        logger.exception(f"API error creating project: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>', methods=['GET'])
def api_project(project_id):
    """API: Get project details"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if request.headers.get('X-API-Key'):
            if not api_key_check_project_access(project):
                return jsonify({'error': 'Permission denied'}), 403
        elif not current_user.is_authenticated and not project.is_public:
            return jsonify({'error': 'Authentication required'}), 401
        elif current_user.is_authenticated and not current_user.is_admin and project.user_id != current_user.id and not project.is_public:
            return jsonify({'error': 'Permission denied'}), 403
    
    return jsonify(project.to_dict())


@app.route('/api/projects/<int:project_id>/status', methods=['GET'])
def api_project_status(project_id):
    """API: Get project status"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if request.headers.get('X-API-Key'):
            if not api_key_check_project_access(project):
                return jsonify({'error': 'Permission denied'}), 403
        elif not current_user.is_authenticated and not project.is_public:
            return jsonify({'error': 'Authentication required'}), 401
        elif current_user.is_authenticated and not current_user.is_admin and project.user_id != current_user.id and not project.is_public:
            return jsonify({'error': 'Permission denied'}), 403
    
    # Check if this project has an active job
    output = []
    if project.directory in active_jobs:
        active_job = active_jobs[project.directory]
        project.status = active_job['status']
        
        if 'cleaned_up' not in active_job:
            output = active_job.get('output', [])  # Fixed indentation here
        
        return jsonify({
            'status': project.status,
            'output': output,
            'start_time': active_job['start_time'].isoformat() if 'start_time' in active_job else None,
            'error': active_job.get('error', None)
        })
    else:
        # Get the latest job from database
        latest_job = JobHistory.query.filter_by(project_id=project_id).order_by(JobHistory.start_time.desc()).first()
        
        if latest_job:
            return jsonify({
                'status': latest_job.status,
                'start_time': latest_job.start_time.isoformat() if latest_job.start_time else None,
                'end_time': latest_job.end_time.isoformat() if latest_job.end_time else None,
                'error': latest_job.error
            })
        
        return jsonify({
            'status': project.status,
            'output': [],
            'error': None
        })


@app.route('/api/projects/<int:project_id>/files', methods=['GET'])
def api_project_files(project_id):
    """API: Get project files"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if request.headers.get('X-API-Key'):
            if not api_key_check_project_access(project):
                return jsonify({'error': 'Permission denied'}), 403
        elif not current_user.is_authenticated and not project.is_public:
            return jsonify({'error': 'Authentication required'}), 401
        elif current_user.is_authenticated and not current_user.is_admin and project.user_id != current_user.id and not project.is_public:
            return jsonify({'error': 'Permission denied'}), 403
    
    project_dir = os.path.join(app.config['PROJECT_FOLDER'], project.directory)
    workspace_dir = os.path.join(project_dir, 'workspace')
    
    if not os.path.exists(workspace_dir):
        return jsonify({'files': []})
    
    files = []
    for root, _, filenames in os.walk(workspace_dir):
        for file in filenames:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, workspace_dir)
            
            files.append({
                'name': rel_path,
                'size': os.path.getsize(file_path),
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return jsonify({'files': files})


@app.route('/api/projects/<int:project_id>/file/<path:file_path>', methods=['GET'])
def api_project_file(project_id, file_path):
    """API: Get file content"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if request.headers.get('X-API-Key'):
            if not api_key_check_project_access(project):
                return jsonify({'error': 'Permission denied'}), 403
        elif not current_user.is_authenticated and not project.is_public:
            return jsonify({'error': 'Authentication required'}), 401
        elif current_user.is_authenticated and not current_user.is_admin and project.user_id != current_user.id and not project.is_public:
            return jsonify({'error': 'Permission denied'}), 403
    
    project_dir = os.path.join(app.config['PROJECT_FOLDER'], project.directory)
    file_full_path = os.path.join(project_dir, 'workspace', file_path)
    
    if not os.path.exists(file_full_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        with open(file_full_path, 'r') as f:
            content = f.read()
        
        return jsonify({
            'content': content,
            'path': file_path
        })
    except UnicodeDecodeError:
        return jsonify({'error': 'Binary file cannot be displayed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/improve', methods=['POST'])
def api_improve_project(project_id):
    """API: Improve a project"""
    if app.config['ENABLE_AUTHENTICATION']:
        if request.headers.get('X-API-Key'):
            return api_key_required(lambda: api_improve_project_impl(project_id))()
        elif not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
    
    return api_improve_project_impl(project_id)


def api_improve_project_impl(project_id):
    """Implementation for API improve project endpoint"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if hasattr(g, 'current_user'):
            if not g.current_user.is_admin and project.user_id != g.current_user.id:
                return jsonify({'error': 'Permission denied'}), 403
        elif current_user.is_authenticated:
            if not current_user.is_admin and project.user_id != current_user.id:
                return jsonify({'error': 'Permission denied'}), 403
    
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt is required'}), 400
    
    project_dir = os.path.join(app.config['PROJECT_FOLDER'], project.directory)
    
    try:
        # Write prompt file
        with open(os.path.join(project_dir, 'prompt'), 'w') as f:
            f.write(data['prompt'])
        
        # Update project in database
        project.prompt = data['prompt']
        project.model = data.get('model', 'gpt-4o')
        project.status = 'pending'
        project.last_modified = datetime.utcnow()
        db.session.commit()
        
        # Gather metadata for job
        metadata = {
            'temperature': float(data.get('temperature', 0.7)),
        }
        
        if 'context_window' in data:
            metadata['context_window'] = int(data['context_window'])
            
        if 'steps' in data:
            metadata['steps'] = data['steps']
        
        # Start GPT-Engineer process in a separate thread
        user_id = None
        if app.config['ENABLE_AUTHENTICATION']:
            if hasattr(g, 'current_user'):
                user_id = g.current_user.id
            elif current_user.is_authenticated:
                user_id = current_user.id
                
        thread = threading.Thread(
            target=run_gpt_engineer,
            args=(project_dir, data.get('model', 'gpt-4o'), "improve"),
            kwargs={
                'metadata': metadata,
                'user_id': user_id
            }
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Project improvement started',
            'project': project.to_dict()
        })
        
    except Exception as e:
        logger.exception(f"API error improving project: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Batch operations API
@app.route('/api/batch/status', methods=['POST'])
def api_batch_status():
    """API: Get status for multiple projects"""
    if app.config['ENABLE_AUTHENTICATION']:
        if request.headers.get('X-API-Key'):
            return api_key_required(lambda: api_batch_status_impl())()
        elif not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
    
    return api_batch_status_impl()


def api_batch_status_impl():
    """Implementation for API batch status endpoint"""
    data = request.json
    
    if not data or 'project_ids' not in data:
        return jsonify({'error': 'Project IDs are required'}), 400
    
    statuses = {}
    
    for project_id in data['project_ids']:
        try:
            project_id = int(project_id)
            project = Project.query.get(project_id)
            
            if not project:
                statuses[project_id] = {'error': 'Project not found'}
                continue
            
            # Check access permissions
            if app.config['ENABLE_AUTHENTICATION']:
                if hasattr(g, 'current_user') and not g.current_user.is_admin and project.user_id != g.current_user.id and not project.is_public:
                    statuses[project_id] = {'error': 'Permission denied'}
                    continue
                elif current_user.is_authenticated and not current_user.is_admin and project.user_id != current_user.id and not project.is_public:
                    statuses[project_id] = {'error': 'Permission denied'}
                    continue
                elif not hasattr(g, 'current_user') and not current_user.is_authenticated and not project.is_public:
                    statuses[project_id] = {'error': 'Permission denied'}
                    continue
            
            # Get status
            if project.directory in active_jobs:
                active_job = active_jobs[project.directory]
                status = {
                    'status': active_job['status'],
                    'start_time': active_job['start_time'].isoformat() if 'start_time' in active_job else None,
                    'error': active_job.get('error', None)
                }
            else:
                # Get the latest job from database
                latest_job = JobHistory.query.filter_by(project_id=project_id).order_by(JobHistory.start_time.desc()).first()
                
                if latest_job:
                    status = {
                        'status': latest_job.status,
                        'start_time': latest_job.start_time.isoformat() if latest_job.start_time else None,
                        'end_time': latest_job.end_time.isoformat() if latest_job.end_time else None,
                        'error': latest_job.error
                    }
                else:
                    status = {
                        'status': project.status,
                        'error': None
                    }
            
            statuses[project_id] = status
            
        except Exception as e:
            logger.exception(f"Error getting status for project {project_id}: {str(e)}")
            statuses[project_id] = {'error': str(e)}
    
    return jsonify({'statuses': statuses})


# Utility routes
@app.route('/project/<int:project_id>/file/<path:file_path>')
def get_file(project_id, file_path):
    """Get file content"""
    project = Project.query.get_or_404(project_id)
    
    # Check access permissions
    if app.config['ENABLE_AUTHENTICATION']:
        if not current_user.is_authenticated and not project.is_public:
            return jsonify({'error': 'Authentication required'}), 401
        elif current_user.is_authenticated and not current_user.is_admin and project.user_id != current_user.id and not project.is_public:
            return jsonify({'error': 'Permission denied'}), 403
    
    project_dir = os.path.join(app.config['PROJECT_FOLDER'], project.directory)
    file_full_path = os.path.join(project_dir, 'workspace', file_path)
    
    if not os.path.exists(file_full_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        with open(file_full_path, 'r') as f:
            content = f.read()
        
        return jsonify({
            'content': content,
            'path': file_path
        })
    except UnicodeDecodeError:
        return jsonify({'error': 'Binary file cannot be displayed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['GET'])
def search_projects():
    """Search for projects"""
    query = request.args.get('q', '').strip()
    
    if not query:
        return redirect(url_for('index'))
    
    # Basic search in project name, description, and tags
    if app.config['ENABLE_AUTHENTICATION']:
        if current_user.is_authenticated:
            projects = Project.query.filter(
                (Project.user_id == current_user.id) | (Project.is_public == True)
            ).filter(
                db.or_(
                    Project.name.ilike(f'%{query}%'),
                    Project.description.ilike(f'%{query}%'),
                    Project.tags.ilike(f'%{query}%'),
                    Project.prompt.ilike(f'%{query}%')
                )
            ).all()
        else:
            projects = Project.query.filter_by(is_public=True).filter(
                db.or_(
                    Project.name.ilike(f'%{query}%'),
                    Project.description.ilike(f'%{query}%'),
                    Project.tags.ilike(f'%{query}%'),
                    Project.prompt.ilike(f'%{query}%')
                )
            ).all()
    else:
        projects = Project.query.filter(
            db.or_(
                Project.name.ilike(f'%{query}%'),
                Project.description.ilike(f'%{query}%'),
                Project.tags.ilike(f'%{query}%'),
                Project.prompt.ilike(f'%{query}%')
            )
        ).all()
    
    return render_template('search_results.html', projects=projects, query=query)


@app.route('/tags/<tag>')
def projects_by_tag(tag):
    """List projects with a specific tag"""
    if app.config['ENABLE_AUTHENTICATION']:
        if current_user.is_authenticated:
            projects = Project.query.filter(
                (Project.user_id == current_user.id) | (Project.is_public == True)
            ).filter(
                Project.tags.ilike(f'%{tag}%')
            ).all()
        else:
            projects = Project.query.filter_by(is_public=True).filter(
                Project.tags.ilike(f'%{tag}%')
            ).all()
    else:
        projects = Project.query.filter(
            Project.tags.ilike(f'%{tag}%')
        ).all()
    
    return render_template('tag_projects.html', projects=projects, tag=tag)


# Helper function for model descriptions
def get_model_description(model_id):
    """Get description for a GPT model"""
    descriptions = {
        'gpt-4o': 'Most powerful model, best for complex tasks',
        'gpt-4-turbo': 'Powerful and efficient for most tasks',
        'gpt-3.5-turbo': 'Fast and cost-effective for simpler tasks',
    }
    
    return descriptions.get(model_id, 'AI language model')


# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code=404, error_message='Page not found'), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception("Server error")
    return render_template('error.html', error_code=500, error_message='Server error'), 500


# Initialize scheduled tasks
init_scheduler()


if __name__ == '__main__':
    # Enable hot reloading in development mode
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=int(os.getenv('PORT', 8080)))

# OpenMind: Government AI with Uncompromising Accuracy

**OpenMind** is an accuracy-first conversational AI system designed specifically for government agencies. It enables natural language querying of government datasets while maintaining the high accuracy and auditability standards required for official decision-making.

## 🎯 Project Overview

OpenMind flips the traditional AI paradigm from "reasoning first" to "accuracy first" through:

- **Dataset-bounded AI** that only responds about available data
- **Multi-layer trust validation** with configurable accuracy thresholds
- **Complete source attribution** for every AI response
- **Scope-limited responses** to prevent hallucination about unavailable data
- **Dynamic database routing** for multi-dataset environments

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (recommended: use pyenv or similar)
- **uv** package manager (for dependency management)
- **OpenAI API Key** (for AI functionality)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd nt-gov-hack
   ```
2. **Install dependencies using uv**

   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project dependencies
   uv sync
   ```
3. **Set up environment variables**

   ```bash
   # Create .env file in project root
   touch .env
   ```

   Add the following to your `.env` file:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DJANGO_SECRET_KEY=your_django_secret_key_here
   DEBUG=True
   ```
4. **Set up the Django application**

   ```bash
   # Navigate to Django project
   cd gov_hack_project

   # Run migrations
   uv run python manage.py makemigrations
   uv run python manage.py migrate

   # Create superuser (optional)
   uv run python manage.py createsuperuser
   ```
5. **Initialize sample databases** (optional)

   ```bash
   # Return to project root
   cd ..

   # The system will automatically create databases from uploaded CSV/Excel files
   # Sample datasets are available in the dataset/ directory
   ```

### Running the Application

1. **Start the Django development server**

   ```bash
   cd gov_hack_project
   uv run python manage.py runserver
   ```
2. **Access the application**

   - Open your browser and navigate to: `http://127.0.0.1:8000`
   - Register a new account or login with existing credentials
   - Upload datasets and start querying with natural language

## 📊 Features

### Core Functionality

- **Natural Language Querying**: Ask questions about your data in plain English
- **Multi-Dataset Support**: Automatic routing between budget, employee, and custom datasets
- **File Upload Processing**: Automatic conversion of CSV/Excel files to queryable databases
- **User Management**: Secure authentication with dataset access controls
- **Audit Logging**: Complete tracking of all queries and data access

### AI Pipeline Features

- **Relevance Checking**: Validates queries are relevant to available datasets (0.7 threshold)
- **Accuracy Validation**: Ensures response accuracy before delivery (0.8 threshold)
- **Reasoning Analysis**: Provides transparent reasoning for AI responses
- **Source Attribution**: Links every claim to specific data records
- **Retry Mechanisms**: Automatic retry with quality control at each stage

## 🗂️ Project Structure

```
nt-gov-hack/
├── gov_hack_project/          # Django web application
│   ├── chat/                  # Main chat application
│   │   ├── models.py         # Dataset and user models
│   │   ├── views.py          # Web interface logic
│   │   ├── templates/        # HTML templates
│   │   └── migrations/       # Database migrations
│   ├── gov_hack_project/     # Django settings
│   ├── manage.py             # Django management script
│   └── db.sqlite3           # Django application database
├── pipeline.py               # LangGraph AI processing pipeline
├── inference_router.py       # Database routing system
├── inference.py              # Basic inference functionality
├── dataset/                  # Sample datasets
├── database_with_details.yaml # Database configuration
└── *.db                      # SQLite databases (auto-generated)
```

## 🔧 Configuration

### Database Configuration

The system uses YAML configuration files to manage multiple databases:

- `database_with_details.yaml` - Main database configuration
- Automatic SQLite database creation from uploaded files
- Dynamic agent creation for each dataset

### AI Pipeline Configuration

Key configurable parameters in `pipeline.py`:

- `RELEVANCE_THRESHOLD = 0.7` - Minimum relevance score for query acceptance
- `ACCURACY_THRESHOLD = 0.8` - Minimum accuracy score for response delivery
- `MAX_RETRIES = 2` - Maximum retry attempts for quality control

## 📝 Usage Examples

### Uploading Datasets

1. Navigate to the Dataset Management interface
2. Upload CSV or Excel files
3. The system automatically converts files to SQLite databases
4. AI agents are dynamically created for each dataset

### Querying Data

Examples of natural language queries:

- **Budget queries**: "What's our total budget for this year?"
- **Employee queries**: "How many employees are in the IT department?"
- **Leave analysis**: "What are the leave patterns in Q3?"
- **Performance analysis**: "Which departments have the highest performance scores?"

### Quality Control

- All responses include confidence scores and source attribution
- Failed relevance/accuracy checks trigger automatic retries
- Complete audit logs are maintained for all interactions

## 🛠️ Development

### Running Tests

```bash
cd gov_hack_project
uv run python manage.py test
```

### Database Management

```bash
# Create new migrations after model changes
uv run python manage.py makemigrations

# Apply migrations
uv run python manage.py migrate

# Access Django admin
uv run python manage.py createsuperuser
# Then visit http://127.0.0.1:8000/admin
```

### Pipeline Testing

```bash
# Test the AI pipeline directly
cd ..
uv run python -c "from pipeline import stream_graph_updates; print(stream_graph_updates('How many employees are there?'))"
```

## 🔒 Security Features

- **User Authentication**: Secure login/registration system
- **Dataset Access Controls**: Owner-based permissions with sharing capabilities
- **Audit Logging**: Complete tracking of all data access and queries
- **Input Validation**: Comprehensive validation of user inputs and file uploads
- **API Key Management**: Secure handling of external API credentials

## 🎯 Government Use Cases

- **Finance Department**: Budget analysis, expense tracking, procurement oversight
- **HR Department**: Employee analytics, leave pattern analysis, performance tracking
- **Operations**: Data anomaly detection, trend analysis, compliance monitoring

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**

   - Ensure your `.env` file contains a valid `OPENAI_API_KEY`
   - Check API key permissions and usage limits
2. **Database Connection Issues**

   - Verify database files exist in the project root
   - Check file permissions for SQLite databases
3. **File Upload Issues**

   - Ensure media directory has write permissions
   - Check file format compatibility (CSV/Excel supported)
4. **Pipeline Import Errors**

   - Verify all dependencies are installed with `uv sync`
   - Check Python path configuration in Django settings

### Logs and Debugging

- Django logs: Check console output when running the development server
- AI pipeline logs: Enable verbose mode in `pipeline.py` for detailed processing logs
- Database query logs: SQL queries are logged in the AI pipeline output

## 🚀 Deployment Notes

For production deployment:

1. Set `DEBUG=False` in environment variables
2. Configure proper database backends (PostgreSQL recommended)
3. Set up proper static file serving
4. Configure HTTPS and security headers
5. Set up proper logging and monitoring

## 📋 Dependencies

Main dependencies (managed via `pyproject.toml`):

- **Django 5.2.5+** - Web framework
- **LangChain 0.3.27+** - AI framework
- **LangGraph 0.6.6+** - AI pipeline orchestration
- **OpenAI** - AI model access
- **Pandas** - Data processing
- **SQLAlchemy** - Database toolkit
- **Python-dotenv** - Environment management

## 🏆 Hackathon Achievement

This project was developed for the **NT Government Hackathon 2024** to demonstrate how government agencies can deploy trustworthy AI systems that meet the stringent accuracy and auditability requirements of public sector decision-making.

## 📞 Support

For questions about setup or usage, please refer to the troubleshooting section above or check the inline documentation in the source code.

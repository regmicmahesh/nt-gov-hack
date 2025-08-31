from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth.views import LoginView
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.db.models import Q
import json
import os
import pandas as pd
import yaml
from sqlalchemy import create_engine
from .forms import UserRegistrationForm, UserLoginForm, DatasetUploadForm, DatasetEditForm, DatasetSearchForm, DatabaseConnectionForm, APIDatasetForm, ExternalLinkForm
from .models import Dataset, DatasetAccessLog
from django.utils import timezone
import sys
import io
import subprocess
from contextlib import redirect_stdout, redirect_stderr

# Add the parent directory to Python path to import pipeline
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)




def get_pipeline_response(user_message):
    """
    Get AI response using the LangGraph pipeline from pipeline.py
    """
    try:
        # Check if required environment variables are set
        import os
        if not os.environ.get('OPENAI_API_KEY'):
            return ("🔑 API Configuration Required: Please set your OPENAI_API_KEY environment variable. "
                   "Create a .env file in the project root with: OPENAI_API_KEY=your_api_key_here")
        
        # Import the pipeline functions
        from pipeline import stream_graph_updates
        
        # The stream_graph_updates function now returns a clean synthesized response
        try:
            # Capture debug output but get clean response
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                clean_response = stream_graph_updates(user_message)
            
            # Return the synthesized response
            if clean_response and clean_response.strip():
                return clean_response
            else:
                return "I processed your query, but I'm having trouble generating a response. Please try rephrasing your question."
                
        except Exception as e:
            return f"Error processing query: {str(e)}"
                
    except ImportError as e:
        return f"Pipeline import error: {str(e)}. Please ensure pipeline.py is available."
    except Exception as e:
        return f"Error in pipeline processing: {str(e)}"

def home(request):
    """Home page view"""
    return render(request, 'chat/home.html')

def register(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Log the user in after successful registration
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            messages.success(request, f'Account created successfully for {username}!')
            return redirect('chat:dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'chat/register.html', {'form': form})

class UserLoginView(LoginView):
    """User login view using class-based view"""
    form_class = UserLoginForm
    template_name = 'chat/login.html'
    redirect_authenticated_user = True
    
    def get_success_url(self):
        return reverse_lazy('chat:dashboard')
    
    def form_invalid(self, form):
        messages.error(self.request, 'Invalid username or password.')
        return super().form_invalid(form)

def user_logout(request):
    """User logout view"""
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    return redirect('chat:home')

@login_required
def dashboard(request):
    """Dashboard view - requires authentication"""
    # Get user's datasets and recent activity
    user_datasets = Dataset.objects.filter(
        Q(owner=request.user) | Q(shared_with=request.user) | Q(is_public=True)
    ).distinct()[:5]
    
    # Get recent access logs
    recent_activity = DatasetAccessLog.objects.filter(
        dataset__in=user_datasets
    ).order_by('-timestamp')[:10]
    
    context = {
        'user': request.user,
        'total_users': User.objects.count(),
        'user_datasets': user_datasets,
        'recent_activity': recent_activity,
        'total_datasets': Dataset.objects.count(),
    }
    return render(request, 'chat/dashboard.html', context)

@login_required
def profile(request):
    """User profile view - requires authentication"""
    if request.method == 'POST':
        user = request.user
        user.first_name = request.POST.get('first_name', user.first_name)
        user.last_name = request.POST.get('last_name', user.last_name)
        user.email = request.POST.get('email', user.email)
        user.save()
        messages.success(request, 'Profile updated successfully!')
        return redirect('chat:profile')
    
    return render(request, 'chat/profile.html', {'user': request.user})

@login_required
def change_password(request):
    """Change password view - requires authentication"""
    if request.method == 'POST':
        user = request.user
        current_password = request.POST.get('current_password')
        new_password1 = request.POST.get('new_password1')
        new_password2 = request.POST.get('new_password2')
        
        if not user.check_password(current_password):
            messages.error(request, 'Current password is incorrect.')
        elif new_password1 != new_password2:
            messages.error(request, 'New passwords do not match.')
        elif len(new_password1) < 8:
            messages.error(request, 'New password must be at least 8 characters long.')
        else:
            user.set_password(new_password1)
            user.save()
            messages.success(request, 'Password changed successfully! Please log in again.')
            return redirect('chat:login')
    
    return render(request, 'chat/change_password.html')

@login_required
def ai_chat(request):
    """AI Chat interface for database queries"""
    return render(request, 'chat/ai_chat.html')

@csrf_exempt
@login_required
def process_chat_message(request):
    """Process chat messages and return AI responses"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            
            if not user_message:
                return JsonResponse({'error': 'No message provided'}, status=400)
            
            # Use the LangGraph pipeline to get AI response
            try:
                ai_response = get_pipeline_response(user_message)
                # Add simple confirmation that query was processed
                ai_response += f"\n\n*Query processed successfully using AI-powered database analysis.*"
            except Exception as e:
                ai_response = f"I apologize, but I encountered an error while processing your query: {str(e)}. Please try rephrasing your question."
            
            return JsonResponse({
                'response': ai_response,
                'timestamp': '2024-08-30 06:50:00',
                'datasets_queried': []
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

# Dataset Management Views

@login_required
def dataset_list(request):
    """List all datasets available to the user"""
    search_form = DatasetSearchForm(request.GET)
    
    # Get datasets user can access
    datasets = Dataset.objects.filter(
        Q(owner=request.user) | Q(shared_with=request.user) | Q(is_public=True)
    ).distinct()
    
    # Apply search filters
    if search_form.is_valid():
        search = search_form.cleaned_data.get('search')
        dataset_type = search_form.cleaned_data.get('dataset_type')
        owner = search_form.cleaned_data.get('owner')
        is_public = search_form.cleaned_data.get('is_public')
        
        if search:
            datasets = datasets.filter(
                Q(name__icontains=search) | 
                Q(description__icontains=search) | 
                Q(tags__icontains=search)
            )
        
        if dataset_type:
            datasets = datasets.filter(dataset_type=dataset_type)
        
        if owner:
            datasets = datasets.filter(owner__username=owner)
        
        if is_public:
            datasets = datasets.filter(is_public=(is_public == 'True'))
    
    # Pagination
    paginator = Paginator(datasets, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_form': search_form,
        'total_datasets': datasets.count(),
    }
    return render(request, 'chat/dataset_list.html', context)

@csrf_exempt
@login_required
def dataset_upload(request):
    """Upload a new dataset"""
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    if request.method == 'POST':
        dataset_source = request.POST.get('dataset_source', 'file')
        
        if dataset_source == 'file':
            form = DatasetUploadForm(request.POST, request.FILES)
        elif dataset_source == 'database':
            form = DatabaseConnectionForm(request.POST)
        elif dataset_source == 'api':
            form = APIDatasetForm(request.POST)
        elif dataset_source == 'external_link':
            form = ExternalLinkForm(request.POST)
        else:
            form = DatasetUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.owner = request.user
            dataset.dataset_source = dataset_source
            
            try:
                if dataset_source == 'file':
                    # Process the uploaded file and create SQLite database
                    csv_file = request.FILES['csv_file']
                    
                    # Read the CSV file
                    if csv_file.name.endswith('.csv'):
                        df = pd.read_csv(csv_file)
                    elif csv_file.name.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(csv_file)
                    else:
                        raise ValueError('Unsupported file format. Please upload CSV or Excel files.')
                    
                    # Create SQLite database filename based on dataset name
                    db_name = f"{dataset.name.lower().replace(' ', '_').replace('-', '_')}.db"
                    
                    # Get the project root directory (parent of gov_hack_project)
                    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    db_path = os.path.join(current_dir, db_name)
                    table_name = dataset.name.lower().replace(' ', '_').replace('-', '_')
                    
                    # Create SQLite database from DataFrame
                    engine = create_engine(f"sqlite:///{db_path}")
                    df.to_sql(table_name, engine, index=False, if_exists='replace')
                    
                    # Update dataset metadata
                    dataset.row_count = len(df)
                    dataset.column_count = len(df.columns)
                    dataset.status = 'active'
                    
                    # Update database.yaml file
                    database_yaml_path = os.path.join(current_dir, 'database.yaml')
                    
                    # Load existing database.yaml
                    try:
                        with open(database_yaml_path, 'r') as file:
                            yaml_data = yaml.safe_load(file) or {}
                    except FileNotFoundError:
                        yaml_data = {}
                    
                    # Initialize databases if it doesn't exist or is an empty list
                    if 'databases' not in yaml_data or isinstance(yaml_data.get('databases'), list):
                        yaml_data['databases'] = {}
                    
                    # Create description with column information
                    columns_info = []
                    for col in df.columns:
                        col_type = str(df[col].dtype)
                        null_count = df[col].isnull().sum()
                        unique_count = df[col].nunique()
                        columns_info.append(f"{col} ({col_type}): {unique_count} unique values, {null_count} nulls")
                    
                    description = f"""This dataset was uploaded from {csv_file.name} and contains {len(df)} rows and {len(df.columns)} columns.
                    
Dataset Type: {dataset.get_dataset_type_display()}
Description: {dataset.description}

Columns:
""" + "\n".join([f"- {col_info}" for col_info in columns_info])
                    
                    # Add new database entry
                    dataset_key = f"{dataset.name.lower().replace(' ', '_')}_dataset"
                    yaml_data['databases'][dataset_key] = {
                        'type': 'database',
                        'file_name': f'sqlite:///{db_name}',
                        'description': description,
                        'table_name': table_name,
                        'created_from': csv_file.name,
                        'owner': request.user.username,
                        'dataset_type': dataset.dataset_type,
                        'row_count': len(df),
                        'column_count': len(df.columns)
                    }
                    
                    # Write updated database.yaml
                    with open(database_yaml_path, 'w') as file:
                        yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False, indent=2)
                    
                    print(f"✅ Created SQLite database: {db_path}")
                    print(f"✅ Updated database.yaml with entry: {dataset_key}")
                    print(f"✅ Database table name: {table_name}")
                    print(f"✅ Database contains {len(df)} rows and {len(df.columns)} columns")
                
                else:
                    # For other source types, just save without processing
                    dataset.status = 'active'
                    if dataset_source in ['database', 'api', 'external_link']:
                        dataset.last_sync = timezone.now()
                
                dataset.save()
                
                # Generate detailed schema information using put_details.py
                if dataset_source == 'file':
                    try:
                        print(f"🔍 Running put_details.py to generate detailed schema information...")
                        
                        # Run put_details.py as subprocess from the project root
                        put_details_path = os.path.join(current_dir, 'put_details.py')
                        
                        if os.path.exists(put_details_path):
                            # Change to project root directory and run put_details.py
                            result = subprocess.run(
                                [sys.executable, 'put_details.py'], 
                                cwd=current_dir,
                                capture_output=True,
                                text=True,
                                timeout=120  # 2 minute timeout
                            )
                            
                            if result.returncode == 0:
                                print(f"✅ Generated detailed schema information successfully")
                                if result.stdout:
                                    print(f"📝 Output: {result.stdout[:200]}...")
                            else:
                                print(f"⚠️  put_details.py returned error code {result.returncode}")
                                if result.stderr:
                                    print(f"❌ Error: {result.stderr[:200]}...")
                        else:
                            print(f"⚠️  put_details.py not found at {put_details_path}")
                            
                    except subprocess.TimeoutExpired:
                        print(f"⚠️  put_details.py timed out after 2 minutes")
                    except Exception as schema_error:
                        print(f"⚠️  Warning: Could not run put_details.py: {schema_error}")
                    # Don't fail the upload if schema generation fails
                
                # Log the creation
                DatasetAccessLog.objects.create(
                    dataset=dataset,
                    user=request.user,
                    access_type='view',
                    result_summary=f'Dataset created from {dataset_source} source'
                )
                
                if is_ajax:
                    return JsonResponse({
                        'success': True,
                        'message': f'Dataset "{dataset.name}" created successfully!',
                        'dataset_id': dataset.id,
                        'redirect_url': f'/datasets/{dataset.id}/'
                    })
                else:
                    messages.success(request, f'Dataset "{dataset.name}" created successfully!')
                    return redirect('chat:dataset_detail', dataset_id=dataset.id)
                
            except Exception as e:
                if is_ajax:
                    return JsonResponse({
                        'success': False,
                        'error': f'Error processing dataset: {str(e)}'
                    }, status=400)
                else:
                    dataset.status = 'error'
                    dataset.processing_errors = str(e)
                    dataset.save()
                    messages.error(request, f'Error processing dataset: {str(e)}')
        else:
            # Form validation failed
            if is_ajax:
                # Return form errors as JSON
                errors = {}
                for field, field_errors in form.errors.items():
                    errors[field] = field_errors
                return JsonResponse({
                    'success': False,
                    'error': 'Form validation failed',
                    'form_errors': errors
                }, status=400)
    else:
        form = DatasetUploadForm()
    
    # For non-AJAX requests, return the regular template
    if is_ajax:
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
    
    return render(request, 'chat/dataset_upload.html', {'form': form})

@login_required
def dataset_detail(request, dataset_id):
    """View dataset details"""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check access permissions
    if not (dataset.owner == request.user or request.user in dataset.shared_with.all() or dataset.is_public):
        messages.error(request, 'You do not have access to this dataset.')
        return redirect('chat:dataset_list')
    
    # Log access
    DatasetAccessLog.objects.create(
        dataset=dataset,
        user=request.user,
        access_type='view'
    )
    
    # Update last accessed
    dataset.last_accessed = timezone.now()
    dataset.save()
    
    context = {
        'dataset': dataset,
        'can_edit': dataset.owner == request.user,
        'columns_list': dataset.get_columns_list(),
        'sample_preview': dataset.get_sample_preview(),
        'tags_list': dataset.get_tags_list(),
    }
    return render(request, 'chat/dataset_detail.html', context)

@login_required
def dataset_edit(request, dataset_id):
    """Edit dataset"""
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    
    if request.method == 'POST':
        form = DatasetEditForm(request.POST, instance=dataset)
        if form.is_valid():
            form.save()
            messages.success(request, f'Dataset "{dataset.name}" updated successfully!')
            return redirect('chat:dataset_detail', dataset_id=dataset.id)
    else:
        form = DatasetEditForm(instance=dataset)
    
    return render(request, 'chat/dataset_edit.html', {'form': form, 'dataset': dataset})

@login_required
def dataset_delete(request, dataset_id):
    """Delete dataset"""
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    
    if request.method == 'POST':
        name = dataset.name
        dataset.delete()
        messages.success(request, f'Dataset "{name}" deleted successfully!')
        return redirect('chat:dataset_list')
    
    return render(request, 'chat/dataset_delete.html', {'dataset': dataset})

@login_required
def dataset_download(request, dataset_id):
    """Download dataset CSV file"""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Check access permissions
    if not (dataset.owner == request.user or request.user in dataset.shared_with.all() or dataset.is_public):
        messages.error(request, 'You do not have access to this dataset.')
        return redirect('chat:dataset_list')
    
    # Log download
    DatasetAccessLog.objects.create(
        dataset=dataset,
        user=request.user,
        access_type='download'
    )
    
    # Return file for download
    from django.http import FileResponse
    response = FileResponse(dataset.csv_file, as_attachment=True)
    response['Content-Disposition'] = f'attachment; filename="{dataset.name}.csv"'
    return response

# Test endpoint for debugging AJAX issues
@csrf_exempt
@login_required
def test_ajax_endpoint(request):
    """Test endpoint to verify AJAX JSON responses are working"""
    if request.method == 'POST':
        try:
            # Try to parse JSON data if sent
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                message = data.get('test_message', 'No message provided')
            else:
                message = request.POST.get('test_message', 'No message provided')
            
            return JsonResponse({
                'success': True,
                'message': f'Test successful! Received: {message}',
                'method': request.method,
                'content_type': request.content_type,
                'is_ajax': request.headers.get('X-Requested-With') == 'XMLHttpRequest'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'Only POST method allowed'
    }, status=405)

@csrf_exempt
def test_database_creation(request):
    """Test endpoint to demonstrate SQLite database creation from CSV"""
    try:
        # Create sample data
        sample_data = {
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Department': ['IT', 'HR', 'Finance']
        }
        df = pd.DataFrame(sample_data)
        
        # Get project root directory
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        db_name = "test_sample.db"
        db_path = os.path.join(current_dir, db_name)
        
        # Create temporary database
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql("employees", engine, index=False, if_exists='replace')
        
        # Test querying the database
        from langchain_community.utilities import SQLDatabase
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        result = db.run("SELECT * FROM employees")
        
        return JsonResponse({
            'success': True,
            'message': 'Test database created successfully!',
            'database_path': db_path,
            'table_name': 'employees',
            'rows_created': len(df),
            'query_result': result
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error creating test database: {str(e)}'
        }, status=500)

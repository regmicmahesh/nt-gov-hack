from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .models import Dataset, DatabaseConnection

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True, help_text='Required. Enter a valid email address.')
    first_name = forms.CharField(max_length=30, required=True, help_text='Required.')
    last_name = forms.CharField(max_length=30, required=True, help_text='Required.')
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError('This email address is already in use.')
        return email
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Customize help text and labels
        self.fields['username'].help_text = 'Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.'
        self.fields['password1'].help_text = 'Your password must contain at least 8 characters and can\'t be entirely numeric.'
        self.fields['password2'].help_text = 'Enter the same password as before, for verification.'
        
        # Add Bootstrap classes
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

class UserLoginForm(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'})
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_messages['invalid_login'] = 'Please enter a correct username and password. Note that both fields may be case-sensitive.'

class DatasetUploadForm(forms.ModelForm):
    """Form for uploading new datasets"""
    
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'dataset_type', 'dataset_source', 'csv_file', 'tags', 'is_public']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter dataset name (e.g., Employee Performance Q4 2024)'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe what this dataset contains, its purpose, and any important context...'
            }),
            'dataset_type': forms.Select(attrs={'class': 'form-control'}),
            'dataset_source': forms.Select(attrs={'class': 'form-control'}),
            'csv_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv,.xlsx,.xls'
            }),
            'tags': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter tags separated by commas (e.g., employees, performance, q4)'
            }),
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def clean_csv_file(self):
        csv_file = self.cleaned_data.get('csv_file')
        dataset_source = self.cleaned_data.get('dataset_source')
        
        if dataset_source == 'file' and not csv_file:
            raise ValidationError('File upload is required for file-based datasets.')
        
        if csv_file:
            # Check file size (max 50MB)
            if csv_file.size > 50 * 1024 * 1024:
                raise ValidationError('File size must be under 50MB.')
            
            # Check file extension
            allowed_extensions = ['.csv', '.xlsx', '.xls']
            if not any(csv_file.name.endswith(ext) for ext in allowed_extensions):
                raise ValidationError('Only CSV and Excel files are allowed.')
            
            # Check if file is empty
            if csv_file.size == 0:
                raise ValidationError('File cannot be empty.')
        
        return csv_file
    
    def clean_tags(self):
        tags = self.cleaned_data.get('tags', '')
        if tags:
            # Clean and validate tags
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            if len(tag_list) > 10:
                raise ValidationError('Maximum 10 tags allowed.')
            return ', '.join(tag_list)
        return tags

class DatabaseConnectionForm(forms.ModelForm):
    """Form for database connection datasets"""
    
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'dataset_type', 'db_host', 'db_name', 'db_port', 
                 'db_username', 'db_password', 'db_type', 'db_table', 'db_query', 
                 'sync_frequency', 'tags', 'is_public']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter dataset name (e.g., Production Database)'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe the database and what data it contains...'
            }),
            'dataset_type': forms.Select(attrs={'class': 'form-control'}),
            'db_host': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'localhost or database server address'
            }),
            'db_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Database name'
            }),
            'db_port': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': '5432 (PostgreSQL), 3306 (MySQL), etc.'
            }),
            'db_username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Database username'
            }),
            'db_password': forms.PasswordInput(attrs={
                'class': 'form-control',
                'placeholder': 'Database password'
            }),
            'db_type': forms.Select(attrs={'class': 'form-control'}),
            'db_table': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Table name or view'
            }),
            'db_query': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'SQL query (optional, leave blank to use entire table)'
            }),
            'sync_frequency': forms.Select(attrs={'class': 'form-control'}),
            'tags': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter tags separated by commas'
            }),
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set dataset source to database
        self.initial['dataset_source'] = 'database'
        self.fields['dataset_source'] = forms.CharField(widget=forms.HiddenInput(), initial='database')
    
    def clean(self):
        cleaned_data = super().clean()
        db_host = cleaned_data.get('db_host')
        db_name = cleaned_data.get('db_name')
        db_type = cleaned_data.get('db_type')
        
        if not all([db_host, db_name, db_type]):
            raise ValidationError('Database host, name, and type are required for database connections.')
        
        return cleaned_data

class APIDatasetForm(forms.ModelForm):
    """Form for API-based datasets"""
    
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'dataset_type', 'api_url', 'api_key', 
                 'sync_frequency', 'tags', 'is_public']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter dataset name (e.g., Weather API Data)'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe the API and what data it provides...'
            }),
            'dataset_type': forms.Select(attrs={'class': 'form-control'}),
            'api_url': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://api.example.com/data'
            }),
            'api_key': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'API key or authentication token'
            }),
            'sync_frequency': forms.Select(attrs={'class': 'form-control'}),
            'tags': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter tags separated by commas'
            }),
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set dataset source to API
        self.initial['dataset_source'] = 'api'
        self.fields['dataset_source'] = forms.CharField(widget=forms.HiddenInput(), initial='api')
    
    def clean(self):
        cleaned_data = super().clean()
        api_url = cleaned_data.get('api_url')
        
        if not api_url:
            raise ValidationError('API URL is required for API-based datasets.')
        
        return cleaned_data

class ExternalLinkForm(forms.ModelForm):
    """Form for external link datasets"""
    
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'dataset_type', 'external_url', 
                 'sync_frequency', 'tags', 'is_public']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter dataset name (e.g., Government Data Portal)'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe the external data source...'
            }),
            'dataset_type': forms.Select(attrs={'class': 'form-control'}),
            'external_url': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://data.gov.au/dataset/...'
            }),
            'sync_frequency': forms.Select(attrs={'class': 'form-control'}),
            'tags': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter tags separated by commas'
            }),
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set dataset source to external link
        self.initial['dataset_source'] = 'external_link'
        self.fields['dataset_source'] = forms.CharField(widget=forms.HiddenInput(), initial='external_link')
    
    def clean(self):
        cleaned_data = super().clean()
        external_url = cleaned_data.get('external_url')
        
        if not external_url:
            raise ValidationError('External URL is required for external link datasets.')
        
        return cleaned_data

class DatasetEditForm(forms.ModelForm):
    """Form for editing existing datasets"""
    
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'dataset_type', 'tags', 'is_public', 'sync_frequency']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'dataset_type': forms.Select(attrs={'class': 'form-control'}),
            'tags': forms.TextInput(attrs={'class': 'form-control'}),
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'sync_frequency': forms.Select(attrs={'class': 'form-control'}),
        }
    
    def clean_tags(self):
        tags = self.cleaned_data.get('tags', '')
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            if len(tag_list) > 10:
                raise ValidationError('Maximum 10 tags allowed.')
            return ', '.join(tag_list)
        return tags

class DatasetSearchForm(forms.Form):
    """Form for searching datasets"""
    
    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search datasets by name, description, or tags...'
        })
    )
    
    dataset_type = forms.ChoiceField(
        choices=[('', 'All Types')] + Dataset.DATASET_TYPES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    dataset_source = forms.ChoiceField(
        choices=[('', 'All Sources')] + Dataset.DATASET_SOURCES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    owner = forms.ChoiceField(
        choices=[('', 'All Users')],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    is_public = forms.ChoiceField(
        choices=[
            ('', 'All'),
            ('True', 'Public Only'),
            ('False', 'Private Only')
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dynamically populate owner choices
        from django.contrib.auth.models import User
        user_choices = [('', 'All Users')]
        for user in User.objects.all():
            user_choices.append((user.username, user.username))
        self.fields['owner'].choices = user_choices

class DatabaseConnectionForm(forms.ModelForm):
    """Form for managing database connections"""
    
    class Meta:
        model = DatabaseConnection
        fields = ['name', 'description', 'db_type', 'db_host', 'db_name', 'db_port', 
                 'db_username', 'db_password', 'use_ssl', 'connection_timeout', 
                 'max_connections', 'is_public']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'db_type': forms.Select(attrs={'class': 'form-control'}),
            'db_host': forms.TextInput(attrs={'class': 'form-control'}),
            'db_name': forms.TextInput(attrs={'class': 'form-control'}),
            'db_port': forms.NumberInput(attrs={'class': 'form-control'}),
            'db_username': forms.TextInput(attrs={'class': 'form-control'}),
            'db_password': forms.PasswordInput(attrs={'class': 'form-control'}),
            'use_ssl': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'connection_timeout': forms.NumberInput(attrs={'class': 'form-control'}),
            'max_connections': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        } 
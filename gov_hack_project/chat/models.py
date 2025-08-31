from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os

# Create your models here.

class Dataset(models.Model):
    """Model for storing dataset information"""
    
    DATASET_TYPES = [
        ('employee', 'Employee Data'),
        ('budget', 'Budget Data'),
        ('performance', 'Performance Data'),
        ('financial', 'Financial Data'),
        ('custom', 'Custom Dataset'),
    ]
    
    DATASET_SOURCES = [
        ('file', 'File Upload (CSV/Excel)'),
        ('database', 'Database Connection'),
        ('api', 'API Endpoint'),
        ('external_link', 'External Link'),
    ]
    
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('archived', 'Archived'),
        ('processing', 'Processing'),
        ('error', 'Error'),
        ('connecting', 'Connecting'),
    ]
    
    # Basic Information
    name = models.CharField(max_length=200, help_text="Name of the dataset")
    description = models.TextField(help_text="Detailed description of the dataset")
    dataset_type = models.CharField(max_length=20, choices=DATASET_TYPES, default='custom')
    dataset_source = models.CharField(max_length=20, choices=DATASET_SOURCES, default='file')
    
    # File Information (for file uploads)
    csv_file = models.FileField(upload_to='datasets/', help_text="Upload CSV/Excel file", null=True, blank=True)
    file_size = models.BigIntegerField(help_text="File size in bytes", null=True, blank=True)
    row_count = models.IntegerField(help_text="Number of rows in the dataset", null=True, blank=True)
    column_count = models.IntegerField(help_text="Number of columns in the dataset", null=True, blank=True)
    
    # Database Connection Information (for database connections)
    db_host = models.CharField(max_length=255, help_text="Database host/connection string", null=True, blank=True)
    db_name = models.CharField(max_length=255, help_text="Database name", null=True, blank=True)
    db_port = models.IntegerField(help_text="Database port", null=True, blank=True)
    db_username = models.CharField(max_length=255, help_text="Database username", null=True, blank=True)
    db_password = models.CharField(max_length=255, help_text="Database password (encrypted)", null=True, blank=True)
    db_type = models.CharField(max_length=50, help_text="Database type (PostgreSQL, MySQL, SQLite, etc.)", null=True, blank=True)
    db_table = models.CharField(max_length=255, help_text="Table name or query", null=True, blank=True)
    db_query = models.TextField(help_text="SQL query or table reference", null=True, blank=True)
    
    # API/External Link Information
    api_url = models.URLField(help_text="API endpoint URL", null=True, blank=True)
    api_key = models.CharField(max_length=500, help_text="API key or authentication", null=True, blank=True)
    external_url = models.URLField(help_text="External data source URL", null=True, blank=True)
    
    # Metadata
    columns_info = models.JSONField(default=dict, help_text="Information about columns (names, types, descriptions)")
    sample_data = models.JSONField(default=list, help_text="Sample rows for preview")
    
    # Status and Processing
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    processing_errors = models.TextField(blank=True, help_text="Any errors during processing")
    last_sync = models.DateTimeField(null=True, blank=True, help_text="Last time data was synced from source")
    sync_frequency = models.CharField(max_length=20, choices=[
        ('manual', 'Manual'),
        ('hourly', 'Hourly'),
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly')
    ], default='manual')
    
    # Ownership and Access
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='datasets')
    is_public = models.BooleanField(default=False, help_text="Whether this dataset is public to all users")
    shared_with = models.ManyToManyField(User, blank=True, related_name='shared_datasets')
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    last_accessed = models.DateTimeField(null=True, blank=True)
    
    # Tags for organization
    tags = models.CharField(max_length=500, blank=True, help_text="Comma-separated tags")
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Dataset'
        verbose_name_plural = 'Datasets'
    
    def __str__(self):
        return f"{self.name} ({self.owner.username})"
    
    def get_file_size_mb(self):
        """Return file size in MB"""
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return 0
    
    def get_columns_list(self):
        """Return list of column names"""
        if self.columns_info:
            return list(self.columns_info.keys())
        return []
    
    def get_sample_preview(self, rows=5):
        """Return sample data for preview"""
        if self.sample_data:
            return self.sample_data[:rows]
        return []
    
    def get_tags_list(self):
        """Return list of tags"""
        if self.tags:
            return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
        return []
    
    def get_source_display(self):
        """Get human-readable source description"""
        if self.dataset_source == 'file':
            return f"File: {self.csv_file.name if self.csv_file else 'No file'}"
        elif self.dataset_source == 'database':
            return f"Database: {self.db_type} - {self.db_name}"
        elif self.dataset_source == 'api':
            return f"API: {self.api_url}"
        elif self.dataset_source == 'external_link':
            return f"External: {self.external_url}"
        return "Unknown source"
    
    def get_connection_status(self):
        """Get connection status for database/API sources"""
        if self.dataset_source in ['database', 'api']:
            if self.last_sync:
                return f"Last synced: {self.last_sync.strftime('%Y-%m-%d %H:%M')}"
            else:
                return "Never synced"
        return None
    
    def save(self, *args, **kwargs):
        # Update file size when file is saved
        if self.csv_file:
            self.file_size = self.csv_file.size
        super().save(*args, **kwargs)
    
    def clean(self):
        """Validate dataset configuration based on source type"""
        from django.core.exceptions import ValidationError
        
        if self.dataset_source == 'file' and not self.csv_file:
            raise ValidationError("File upload is required for file-based datasets")
        
        elif self.dataset_source == 'database':
            if not all([self.db_host, self.db_name, self.db_type]):
                raise ValidationError("Database host, name, and type are required for database connections")
        
        elif self.dataset_source == 'api' and not self.api_url:
            raise ValidationError("API URL is required for API-based datasets")
        
        elif self.dataset_source == 'external_link' and not self.external_url:
            raise ValidationError("External URL is required for external link datasets")

class DatasetAccessLog(models.Model):
    """Model for logging dataset access"""
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='access_logs')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    access_type = models.CharField(max_length=20, choices=[
        ('view', 'Viewed'),
        ('query', 'Queried'),
        ('download', 'Downloaded'),
        ('share', 'Shared'),
        ('sync', 'Synced'),
        ('error', 'Error'),
    ])
    timestamp = models.DateTimeField(default=timezone.now)
    query_text = models.TextField(blank=True, help_text="The query that was made")
    result_summary = models.TextField(blank=True, help_text="Summary of the query results")
    error_message = models.TextField(blank=True, help_text="Error message if access failed")
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.user.username} {self.access_type} {self.dataset.name} at {self.timestamp}"

class DatabaseConnection(models.Model):
    """Model for storing reusable database connections"""
    
    name = models.CharField(max_length=200, help_text="Name for this database connection")
    description = models.TextField(help_text="Description of the database")
    db_type = models.CharField(max_length=50, choices=[
        ('postgresql', 'PostgreSQL'),
        ('mysql', 'MySQL'),
        ('sqlite', 'SQLite'),
        ('oracle', 'Oracle'),
        ('sqlserver', 'SQL Server'),
        ('mongodb', 'MongoDB'),
        ('other', 'Other'),
    ])
    db_host = models.CharField(max_length=255, help_text="Database host or connection string")
    db_name = models.CharField(max_length=255, help_text="Database name")
    db_port = models.IntegerField(help_text="Database port", null=True, blank=True)
    db_username = models.CharField(max_length=255, help_text="Database username")
    db_password = models.CharField(max_length=255, help_text="Database password (encrypted)")
    
    # Connection settings
    use_ssl = models.BooleanField(default=False, help_text="Use SSL connection")
    connection_timeout = models.IntegerField(default=30, help_text="Connection timeout in seconds")
    max_connections = models.IntegerField(default=10, help_text="Maximum number of connections")
    
    # Ownership
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='database_connections')
    is_public = models.BooleanField(default=False, help_text="Whether this connection is public")
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    last_tested = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True, help_text="Whether this connection is active")
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Database Connection'
        verbose_name_plural = 'Database Connections'
    
    def __str__(self):
        return f"{self.name} ({self.db_type} - {self.db_host})"
    
    def get_connection_string(self):
        """Get formatted connection string"""
        if self.db_type == 'sqlite':
            return f"sqlite:///{self.db_host}"
        elif self.db_type == 'postgresql':
            port = f":{self.db_port}" if self.db_port else ""
            return f"postgresql://{self.db_username}:{self.db_password}@{self.db_host}{port}/{self.db_name}"
        elif self.db_type == 'mysql':
            port = f":{self.db_port}" if self.db_port else ""
            return f"mysql://{self.db_username}:{self.db_password}@{self.db_host}{port}/{self.db_name}"
        else:
            return f"{self.db_type}://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

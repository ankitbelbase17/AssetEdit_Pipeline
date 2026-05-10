"""
Models for the SI3DR Project Showcase Website

This module defines all database models for storing project information,
team members, technologies, features, screenshots, and contact submissions.
"""

from django.db import models
from django.utils import timezone


class ProjectInfo(models.Model):
    """
    Stores main project information including title, abstract, and problem statement.
    Only one instance should exist in the database.
    """
    title = models.CharField(max_length=300, default="Single Image 3D Reconstruction")
    short_title = models.CharField(max_length=100, default="SI3DR")
    tagline = models.CharField(
        max_length=500,
        default="Generating detailed 3D models from a single 2D image using AI"
    )
    abstract = models.TextField(
        help_text="Main abstract of the project"
    )
    problem_statement = models.TextField(
        help_text="Detailed problem statement"
    )
    institution = models.CharField(
        max_length=300,
        default="Tribhuvan University, Institute of Engineering, Pulchowk Campus"
    )
    department = models.CharField(
        max_length=200,
        default="Department of Electronics and Computer Engineering"
    )
    submission_date = models.CharField(max_length=50, default="December 2025")
    hero_image = models.ImageField(
        upload_to='hero/',
        blank=True,
        null=True,
        help_text="Main hero image for homepage"
    )
    
    class Meta:
        verbose_name = "Project Information"
        verbose_name_plural = "Project Information"
    
    def __str__(self):
        return self.title


class Objective(models.Model):
    """
    Stores individual project objectives.
    """
    title = models.CharField(max_length=200)
    description = models.TextField()
    order = models.PositiveIntegerField(default=0)
    icon = models.CharField(
        max_length=50,
        default="fas fa-check-circle",
        help_text="Font Awesome icon class"
    )
    
    class Meta:
        ordering = ['order']
        verbose_name = "Project Objective"
        verbose_name_plural = "Project Objectives"
    
    def __str__(self):
        return self.title


class TeamMember(models.Model):
    """
    Stores information about team members.
    """
    name = models.CharField(max_length=100)
    roll_number = models.CharField(max_length=50)
    email = models.EmailField(blank=True)
    role = models.CharField(max_length=100, default="Developer")
    photo = models.ImageField(upload_to='team/', blank=True, null=True)
    linkedin_url = models.URLField(blank=True)
    github_url = models.URLField(blank=True)
    order = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['order']
        verbose_name = "Team Member"
        verbose_name_plural = "Team Members"
    
    def __str__(self):
        return f"{self.name} ({self.roll_number})"


class Supervisor(models.Model):
    """
    Stores information about project supervisors.
    """
    name = models.CharField(max_length=100)
    title = models.CharField(max_length=100, help_text="e.g., Assoc. Prof. Dr.")
    role = models.CharField(max_length=100, default="Project Supervisor")
    email = models.EmailField(blank=True)
    photo = models.ImageField(upload_to='supervisors/', blank=True, null=True)
    order = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['order']
        verbose_name = "Supervisor"
        verbose_name_plural = "Supervisors"
    
    def __str__(self):
        return f"{self.title} {self.name}"


class Technology(models.Model):
    """
    Stores technologies used in the project.
    """
    CATEGORY_CHOICES = [
        ('language', 'Programming Language'),
        ('framework', 'Framework/Library'),
        ('tool', 'Tool/Platform'),
        ('model', 'ML Model'),
        ('hardware', 'Hardware'),
    ]
    
    name = models.CharField(max_length=100)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    description = models.TextField(blank=True)
    icon = models.CharField(
        max_length=50,
        blank=True,
        help_text="Font Awesome icon class or custom icon"
    )
    logo = models.ImageField(upload_to='tech_logos/', blank=True, null=True)
    url = models.URLField(blank=True, help_text="Official website URL")
    order = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['category', 'order']
        verbose_name = "Technology"
        verbose_name_plural = "Technologies"
    
    def __str__(self):
        return f"{self.name} ({self.get_category_display()})"


class SystemPhase(models.Model):
    """
    Stores information about different phases of the system architecture.
    """
    title = models.CharField(max_length=200)
    phase_number = models.PositiveIntegerField()
    description = models.TextField()
    details = models.TextField(blank=True, help_text="Additional technical details")
    icon = models.CharField(max_length=50, default="fas fa-cog")
    
    class Meta:
        ordering = ['phase_number']
        verbose_name = "System Phase"
        verbose_name_plural = "System Phases"
    
    def __str__(self):
        return f"Phase {self.phase_number}: {self.title}"


class Feature(models.Model):
    """
    Stores project features and modules.
    """
    title = models.CharField(max_length=200)
    description = models.TextField()
    technical_details = models.TextField(blank=True)
    icon = models.CharField(max_length=50, default="fas fa-star")
    image = models.ImageField(upload_to='features/', blank=True, null=True)
    order = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['order']
        verbose_name = "Feature/Module"
        verbose_name_plural = "Features/Modules"
    
    def __str__(self):
        return self.title


class Screenshot(models.Model):
    """
    Stores project screenshots and demo images.
    """
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='screenshots/')
    category = models.CharField(
        max_length=50,
        default="General",
        help_text="e.g., Reconstruction, Editing, Results"
    )
    order = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['category', 'order']
        verbose_name = "Screenshot"
        verbose_name_plural = "Screenshots"
    
    def __str__(self):
        return self.title


class Result(models.Model):
    """
    Stores project results and outcomes.
    """
    title = models.CharField(max_length=200)
    description = models.TextField()
    metric_value = models.CharField(max_length=50, blank=True)
    metric_label = models.CharField(max_length=100, blank=True)
    icon = models.CharField(max_length=50, default="fas fa-chart-line")
    order = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['order']
        verbose_name = "Result/Outcome"
        verbose_name_plural = "Results/Outcomes"
    
    def __str__(self):
        return self.title


class FutureScope(models.Model):
    """
    Stores future scope and enhancement items.
    """
    CATEGORY_CHOICES = [
        ('reconstruction', 'Reconstruction Pipeline'),
        ('editing', 'Editing Pipeline'),
        ('system', 'System-Level'),
        ('research', 'Research Direction'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    icon = models.CharField(max_length=50, default="fas fa-rocket")
    order = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['category', 'order']
        verbose_name = "Future Scope"
        verbose_name_plural = "Future Scope Items"
    
    def __str__(self):
        return self.title


class ContactSubmission(models.Model):
    """
    Stores contact form submissions.
    """
    name = models.CharField(max_length=100)
    email = models.EmailField()
    subject = models.CharField(max_length=200)
    message = models.TextField()
    submitted_at = models.DateTimeField(default=timezone.now)
    is_read = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-submitted_at']
        verbose_name = "Contact Submission"
        verbose_name_plural = "Contact Submissions"
    
    def __str__(self):
        return f"{self.name} - {self.subject} ({self.submitted_at.strftime('%Y-%m-%d')})"


class Methodology(models.Model):
    """
    Stores methodology/workflow steps.
    """
    step_number = models.PositiveIntegerField()
    title = models.CharField(max_length=200)
    description = models.TextField()
    technical_details = models.TextField(blank=True)
    icon = models.CharField(max_length=50, default="fas fa-arrow-right")
    
    class Meta:
        ordering = ['step_number']
        verbose_name = "Methodology Step"
        verbose_name_plural = "Methodology Steps"
    
    def __str__(self):
        return f"Step {self.step_number}: {self.title}"


class Limitation(models.Model):
    """
    Stores project limitations.
    """
    CATEGORY_CHOICES = [
        ('reconstruction', 'Reconstruction Pipeline'),
        ('editing', 'Editing Pipeline'),
        ('system', 'System-Level'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    order = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['category', 'order']
        verbose_name = "Limitation"
        verbose_name_plural = "Limitations"
    
    def __str__(self):
        return self.title

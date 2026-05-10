"""
Django Admin Configuration for SI3DR Project Showcase

This module configures the Django admin interface for managing all project content.
"""

from django.contrib import admin
from django.utils.html import format_html
from .models import (
    ProjectInfo, Objective, TeamMember, Supervisor, Technology,
    SystemPhase, Feature, Screenshot, Result, FutureScope,
    ContactSubmission, Methodology, Limitation
)


# Customize admin site header
admin.site.site_header = "SI3DR Website Administration"
admin.site.site_title = "SI3DR Admin"
admin.site.index_title = "Single Image 3D Reconstruction - Website Management"


@admin.register(ProjectInfo)
class ProjectInfoAdmin(admin.ModelAdmin):
    """Admin interface for Project Information."""
    list_display = ['title', 'institution', 'submission_date']
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'short_title', 'tagline')
        }),
        ('Project Details', {
            'fields': ('abstract', 'problem_statement')
        }),
        ('Institution', {
            'fields': ('institution', 'department', 'submission_date')
        }),
        ('Media', {
            'fields': ('hero_image',)
        }),
    )


@admin.register(Objective)
class ObjectiveAdmin(admin.ModelAdmin):
    """Admin interface for Project Objectives."""
    list_display = ['title', 'order', 'icon']
    list_editable = ['order']
    ordering = ['order']
    search_fields = ['title', 'description']


@admin.register(TeamMember)
class TeamMemberAdmin(admin.ModelAdmin):
    """Admin interface for Team Members."""
    list_display = ['name', 'roll_number', 'role', 'email', 'order']
    list_editable = ['order']
    search_fields = ['name', 'roll_number', 'email']
    list_filter = ['role']


@admin.register(Supervisor)
class SupervisorAdmin(admin.ModelAdmin):
    """Admin interface for Supervisors."""
    list_display = ['name', 'title', 'role', 'email', 'order']
    list_editable = ['order']
    search_fields = ['name', 'email']


@admin.register(Technology)
class TechnologyAdmin(admin.ModelAdmin):
    """Admin interface for Technologies."""
    list_display = ['name', 'category', 'order']
    list_editable = ['order']
    list_filter = ['category']
    search_fields = ['name', 'description']


@admin.register(SystemPhase)
class SystemPhaseAdmin(admin.ModelAdmin):
    """Admin interface for System Phases."""
    list_display = ['phase_number', 'title']
    ordering = ['phase_number']
    search_fields = ['title', 'description']


@admin.register(Feature)
class FeatureAdmin(admin.ModelAdmin):
    """Admin interface for Features/Modules."""
    list_display = ['title', 'order', 'icon']
    list_editable = ['order']
    ordering = ['order']
    search_fields = ['title', 'description']


@admin.register(Screenshot)
class ScreenshotAdmin(admin.ModelAdmin):
    """Admin interface for Screenshots."""
    list_display = ['title', 'category', 'order', 'image_preview']
    list_editable = ['order']
    list_filter = ['category']
    search_fields = ['title', 'description']
    
    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height: 50px; max-width: 100px;" />',
                obj.image.url
            )
        return "No image"
    image_preview.short_description = "Preview"


@admin.register(Result)
class ResultAdmin(admin.ModelAdmin):
    """Admin interface for Results/Outcomes."""
    list_display = ['title', 'metric_value', 'metric_label', 'order']
    list_editable = ['order']
    ordering = ['order']
    search_fields = ['title', 'description']


@admin.register(FutureScope)
class FutureScopeAdmin(admin.ModelAdmin):
    """Admin interface for Future Scope items."""
    list_display = ['title', 'category', 'order']
    list_editable = ['order']
    list_filter = ['category']
    search_fields = ['title', 'description']


@admin.register(ContactSubmission)
class ContactSubmissionAdmin(admin.ModelAdmin):
    """Admin interface for Contact Submissions."""
    list_display = ['name', 'email', 'subject', 'submitted_at', 'is_read']
    list_filter = ['is_read', 'submitted_at']
    search_fields = ['name', 'email', 'subject', 'message']
    readonly_fields = ['name', 'email', 'subject', 'message', 'submitted_at']
    list_editable = ['is_read']
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return True  # Allow marking as read


@admin.register(Methodology)
class MethodologyAdmin(admin.ModelAdmin):
    """Admin interface for Methodology Steps."""
    list_display = ['step_number', 'title']
    ordering = ['step_number']
    search_fields = ['title', 'description']


@admin.register(Limitation)
class LimitationAdmin(admin.ModelAdmin):
    """Admin interface for Limitations."""
    list_display = ['title', 'category', 'order']
    list_editable = ['order']
    list_filter = ['category']
    search_fields = ['title', 'description']

"""
Forms for the SI3DR Project Showcase Website

This module defines forms for contact submissions and other user input.
"""

from django import forms
from .models import ContactSubmission


class ContactForm(forms.ModelForm):
    """
    Contact form for visitor inquiries.
    """
    
    class Meta:
        model = ContactSubmission
        fields = ['name', 'email', 'subject', 'message']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Your Full Name',
                'required': True,
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'Your Email Address',
                'required': True,
            }),
            'subject': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Subject',
                'required': True,
            }),
            'message': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': 'Your Message',
                'rows': 5,
                'required': True,
            }),
        }
    
    def clean_email(self):
        """Validate email format."""
        email = self.cleaned_data.get('email')
        if email and '@' not in email:
            raise forms.ValidationError('Please enter a valid email address.')
        return email
    
    def clean_message(self):
        """Ensure message has minimum length."""
        message = self.cleaned_data.get('message')
        if message and len(message) < 10:
            raise forms.ValidationError('Message must be at least 10 characters long.')
        return message

"""
ASGI config for SI3DR Website project.

It exposes the ASGI callable as a module-level variable named ``application``.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'si3dr_website.settings')

application = get_asgi_application()

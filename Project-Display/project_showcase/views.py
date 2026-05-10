"""
Views for the SI3DR Project Showcase Website

This module contains all view functions for rendering the promotional website pages
and handling form submissions.
"""

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import json
import base64
import uuid
import subprocess
from .models import (
    ProjectInfo, Objective, TeamMember, Supervisor, Technology,
    SystemPhase, Feature, Screenshot, Result, FutureScope,
    ContactSubmission, Methodology, Limitation
)
from .forms import ContactForm
from .museum_data import get_museum_artifacts


def get_project_context():
    """
    Returns common context data used across all pages.
    """
    try:
        project_info = ProjectInfo.objects.first()
    except ProjectInfo.DoesNotExist:
        project_info = None
    
    return {
        'project_info': project_info,
        'project_title': project_info.title if project_info else "Single Image 3D Reconstruction",
        'institution': project_info.institution if project_info else "Tribhuvan University, IOE Pulchowk Campus",
    }


def home(request):
    """
    Renders the homepage with hero section and project overview.
    """
    context = get_project_context()
    context['objectives'] = Objective.objects.all()[:4]
    context['features'] = Feature.objects.all()[:6]
    context['team_members'] = TeamMember.objects.all()
    context['technologies'] = Technology.objects.all()[:8]
    context['results'] = Result.objects.all()[:4]
    return render(request, 'project_showcase/home.html', context)


def about(request):
    """
    Renders the About Project page with abstract, problem statement, and objectives.
    """
    context = get_project_context()
    context['objectives'] = Objective.objects.all()
    context['team_members'] = TeamMember.objects.all()
    context['supervisors'] = Supervisor.objects.all()
    return render(request, 'project_showcase/about.html', context)


def technology(request):
    """
    Renders the Technology Stack page.
    """
    context = get_project_context()
    
    # Group technologies by category
    context['languages'] = Technology.objects.filter(category='language')
    context['frameworks'] = Technology.objects.filter(category='framework')
    context['tools'] = Technology.objects.filter(category='tool')
    context['models'] = Technology.objects.filter(category='model')
    context['hardware'] = Technology.objects.filter(category='hardware')
    context['all_technologies'] = Technology.objects.all()
    
    return render(request, 'project_showcase/technology.html', context)


def architecture(request):
    """
    Renders the System Architecture page with workflow diagrams.
    """
    context = get_project_context()
    context['phases'] = SystemPhase.objects.all()
    context['methodology_steps'] = Methodology.objects.all()
    return render(request, 'project_showcase/architecture.html', context)


def features(request):
    """
    Renders the Features/Modules page.
    """
    context = get_project_context()
    context['features'] = Feature.objects.all()
    return render(request, 'project_showcase/features.html', context)


def screenshots(request):
    """
    Renders the Screenshots/Demo page with virtual museum gallery.
    """
    context = get_project_context()
    context['screenshots'] = Screenshot.objects.all()
    
    # Group screenshots by category
    categories = Screenshot.objects.values_list('category', flat=True).distinct()
    context['categories'] = categories
    context['screenshots_by_category'] = {
        cat: Screenshot.objects.filter(category=cat)
        for cat in categories
    }
    
    # Load museum artifacts for the gallery
    context['artifacts'] = get_museum_artifacts()
    
    return render(request, 'project_showcase/screenshots.html', context)


def results(request):
    """
    Renders the Results & Outcomes page.
    """
    context = get_project_context()
    context['results'] = Result.objects.all()
    context['limitations'] = Limitation.objects.all()
    
    # Group limitations by category
    context['reconstruction_limitations'] = Limitation.objects.filter(category='reconstruction')
    context['editing_limitations'] = Limitation.objects.filter(category='editing')
    context['system_limitations'] = Limitation.objects.filter(category='system')
    
    # Load museum artifacts for reconstruction results
    context['artifacts'] = get_museum_artifacts()
    
    return render(request, 'project_showcase/results.html', context)


def future_scope(request):
    """
    Renders the Future Scope page with possible enhancements.
    """
    context = get_project_context()
    context['future_items'] = FutureScope.objects.all()
    
    # Group by category
    context['reconstruction_future'] = FutureScope.objects.filter(category='reconstruction')
    context['editing_future'] = FutureScope.objects.filter(category='editing')
    context['system_future'] = FutureScope.objects.filter(category='system')
    context['research_future'] = FutureScope.objects.filter(category='research')
    
    return render(request, 'project_showcase/future_scope.html', context)


def contact(request):
    """
    Renders the Contact page with contact form.
    """
    context = get_project_context()
    
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Thank you for your message! We will get back to you soon.')
            return redirect('project_showcase:contact')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = ContactForm()
    
    context['form'] = form
    context['team_members'] = TeamMember.objects.all()
    context['supervisors'] = Supervisor.objects.all()
    
    return render(request, 'project_showcase/contact.html', context)


@require_POST
@csrf_protect
def contact_submit(request):
    """
    AJAX endpoint for contact form submission.
    """
    form = ContactForm(request.POST)
    if form.is_valid():
        submission = form.save()
        return JsonResponse({
            'success': True,
            'message': 'Thank you for your message! We will get back to you soon.'
        })
    else:
        return JsonResponse({
            'success': False,
            'errors': form.errors
        }, status=400)


def team(request):
    """
    Renders the Team page with team members and supervisors.
    """
    context = get_project_context()
    context['team_members'] = TeamMember.objects.all()
    context['supervisors'] = Supervisor.objects.all()
    return render(request, 'project_showcase/team.html', context)


def virtual_museum(request):
    """
    Renders the Virtual Museum page for visualizing GLB files.
    """
    context = get_project_context()
    # Now perfectly tracks native AND live-generated user submissions
    context['artifacts'] = get_museum_artifacts()
    return render(request, 'project_showcase/virtual_museum.html', context)


def demo(request):
    """
    Renders the Try-It Demo page for 3D reconstruction.
    Users can select from gallery, upload an image, or capture via camera.
    """
    context = get_project_context()
    context['artifacts'] = get_museum_artifacts()
    return render(request, 'project_showcase/demo.html', context)


from .config import HUNYUAN_BACKEND_URL, FRONTEND_PUBLIC_URL
import urllib.request
import urllib.error

@csrf_exempt
@require_POST
def generate_3d(request):
    """
    Sends an async webhook generation request to the remote Backend API.
    Does not block for minutes—instead, receives a session_id and waits for the backend to upload the GLB later.
    """
    try:
        data = json.loads(request.body)
        image_data = data.get('image')
        
        if not image_data:
            return JsonResponse({'success': False, 'error': 'No image data provided'}, status=400)
            
        session_id = str(uuid.uuid4())[:8]
        webhook_url = f"{FRONTEND_PUBLIC_URL.rstrip('/')}/api/webhook/receive/"
        
        # Save local copy for gallery thumbnail embedding
        try:
            if ';base64,' in image_data:
                img_b64 = image_data.split(';base64,')[-1]
                img_bytes = base64.b64decode(img_b64)
                output_dir = os.path.join(settings.BASE_DIR, 'media', 'Outputs')
                os.makedirs(output_dir, exist_ok=True)
                img_path = os.path.join(output_dir, f"asset_image_{session_id}.jpg")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
        except Exception as e:
            print(f"Warning: Could not save local thumbnail: {e}")
        
        print(f"\n[{session_id}] 🚀 Dispatching ASYNC 3D JOB to Remote Backend.")
        print(f"[{session_id}] 🔗 Embedded Webhook Target: {webhook_url}")
        
        payload = {
            'image': image_data,
            'session_id': session_id,
            'webhook_url': webhook_url,
            'job_type': '3d'
        }
        
        backend_endpoint = f"{HUNYUAN_BACKEND_URL.rstrip('/')}/api/generate"
        
        try:
            req = urllib.request.Request(
                backend_endpoint,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            # The backend now responds asynchronously but Cloudflare routing might add overhead latency
            with urllib.request.urlopen(req, timeout=45) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                # If the backend is running asynchronously, it will just return {"success": True, "message": "...", "session_id": "..."}
                return JsonResponse({
                    'success': True,
                    'message': 'Processing started in backend.',
                    'session_id': session_id
                })
                
        except urllib.error.URLError as e:
            return JsonResponse({'success': False, 'error': f'Failed to trigger remote job: {str(e)}'}, status=500)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_POST
def generate_heatmap(request):
    """
    Sends an async webhook heatmap request to the remote backend.
    """
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        
        if not session_id:
            return JsonResponse({'success': False, 'error': 'No session_id provided'}, status=400)
            
        webhook_url = f"{FRONTEND_PUBLIC_URL.rstrip('/')}/api/webhook/receive/"
        
        print(f"\n[{session_id}] 🚀 Dispatching ASYNC HEATMAP JOB to Remote Backend.")
        print(f"[{session_id}] 🔗 Embedded Webhook Target: {webhook_url}")
        
        payload = {
            'session_id': session_id,
            'webhook_url': webhook_url,
            'job_type': 'heatmap'
        }
        
        backend_endpoint = f"{HUNYUAN_BACKEND_URL.rstrip('/')}/api/generate-heatmap"
        
        try:
            req = urllib.request.Request(
                backend_endpoint,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=45) as response:
                result = json.loads(response.read().decode('utf-8'))
                return JsonResponse({
                    'success': True,
                    'message': 'Heatmap calculation started.',
                    'session_id': session_id
                })
                
        except urllib.error.URLError as e:
            return JsonResponse({'success': False, 'error': f'Failed to trigger heatmap: {str(e)}'}, status=500)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
@require_POST
def receive_webhook(request):
    """
    Endpoint where the remote Lightning AI backend pushes AWS JSON pointers natively
    back to this Django frontend.
    """
    try:
        from .config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME, AWS_S3_BUCKET_NAME
        import boto3
        
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = {}
            
        session_id = request.headers.get('X-Session-Id') or data.get('session_id')
        job_type = request.headers.get('X-Job-Type') or data.get('job_type')
        s3_key = data.get('s3_key')
        
        if not all([session_id, job_type, s3_key]):
            return JsonResponse({'success': False, 'error': 'Missing parameters (s3_key, session_id, or job_type)'}, status=400)
            
        print(f"\n[{session_id}] ⚡ INCOMING WEBHOOK! Remote backend replied mapping AWS S3 Pointer: {s3_key}")
        
        output_dir = os.path.join(settings.BASE_DIR, 'media', 'Outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"asset_3d_{session_id}.glb" if job_type == '3d' else f"asset_heatmap_{session_id}.glb"
        filepath = os.path.join(output_dir, filename)
        
        print(f"[{session_id}] 📥 Downloading massive binary natively from AWS '{AWS_S3_BUCKET_NAME}'...")
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION_NAME
        )
        s3_client.download_file(AWS_S3_BUCKET_NAME, s3_key, filepath)
            
        print(f"[{session_id}] ✅ FULLY SAVED TO SITE REPOSITORY DYNAMICALLY: {filepath}\n")
            
        return JsonResponse({"success": True, "message": f"Saved {filename} to local static directory securely from s3 cache"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Webhook Exception: {e}")
        return JsonResponse({"success": False, "error": str(e)}, status=500)


def check_status(request, session_id):
    """
    Polling endpoint triggered every 2 seconds by demo.html.
    It checks if the file has landed. If not, proxies to Remote Backend to fetch live UI progress stats!
    """
    job_type = request.GET.get('type', '3d')
    filename = f"asset_3d_{session_id}.glb" if job_type == '3d' else f"asset_heatmap_{session_id}.glb"
    
    output_dir = os.path.join(settings.BASE_DIR, 'media', 'Outputs')
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        # The background job uploaded the file!
        url = f"{settings.MEDIA_URL}Outputs/{filename}"
        return JsonResponse({"status": "completed", "progress": 100, "message": "Complete!", "url": url})
        
    try:
        # Ask backend for live percentage rendering
        backend_endpoint = f"{HUNYUAN_BACKEND_URL.rstrip('/')}/api/status/{session_id}"
        req = urllib.request.Request(backend_endpoint)
        with urllib.request.urlopen(req, timeout=5) as response:
            backend_data = json.loads(response.read().decode('utf-8'))
            if backend_data.get('success'):
                prog = backend_data['status'].get('progress', 0)
                msg = backend_data['status'].get('message', 'Processing...')
                
                # Check explicitly for crashes reported by the remote background thread
                status_code = "processing"
                if "crash" in msg.lower() or "fail" in msg.lower() or "error" in msg.lower():
                    status_code = "error"
                    
                return JsonResponse({"status": status_code, "progress": prog, "message": msg})
    except Exception as e:
        pass
        
    return JsonResponse({"status": "processing", "progress": 0, "message": "Waiting for backend synchronization..."})

@csrf_exempt
@require_POST
def delete_asset_view(request, session_id):
    """
    Called by the Virtual Museum UI to natively purge user-generated elements from the frontend file system seamlessly!
    """
    try:
        output_dir = os.path.join(settings.BASE_DIR, 'media', 'Outputs')
        files = [
            f"asset_3d_{session_id}.glb",
            f"asset_heatmap_{session_id}.glb",
            f"asset_image_{session_id}.jpg",
            f"image_asset_{session_id}.jpg"
        ]
        
        deleted_count = 0
        for f in files:
            p = os.path.join(output_dir, f)
            if os.path.exists(p):
                os.remove(p)
                deleted_count += 1
                
        return JsonResponse({"success": True, "deleted_count": deleted_count, "message": "Successfully purged from media outputs."})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@csrf_exempt
@require_POST
def delete_museum_asset_view(request, slug):
    """
    Deletes a museum artifact directory and its comparison heatmaps.
    The slug maps to a folder under static/museum_assets/ and static/comparison/.
    """
    import shutil
    
    try:
        static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else os.path.join(settings.BASE_DIR, 'static')
        museum_dir = os.path.join(static_dir, 'museum_assets')
        comparison_dir = os.path.join(static_dir, 'comparison')
        
        deleted_items = []
        
        # Find and delete the museum asset folder
        # The slug uses underscores but folder names may use spaces or mixed case
        if os.path.isdir(museum_dir):
            for folder_name in os.listdir(museum_dir):
                folder_slug = folder_name.lower().replace(' ', '_')
                if folder_slug == slug:
                    target = os.path.join(museum_dir, folder_name)
                    shutil.rmtree(target, ignore_errors=True)
                    deleted_items.append(f"museum_assets/{folder_name}")
                    break
        
        # Find and delete the comparison heatmap folder
        # Comparison dirs may use spaces or underscores
        if os.path.isdir(comparison_dir):
            for folder_name in os.listdir(comparison_dir):
                folder_slug = folder_name.lower().replace(' ', '_')
                if folder_slug == slug:
                    target = os.path.join(comparison_dir, folder_name)
                    shutil.rmtree(target, ignore_errors=True)
                    deleted_items.append(f"comparison/{folder_name}")
                    break
        
        return JsonResponse({
            "success": True,
            "deleted": deleted_items,
            "message": f"Deleted {len(deleted_items)} directories."
        })
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


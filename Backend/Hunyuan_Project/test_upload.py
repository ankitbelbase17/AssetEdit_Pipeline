"""
Test script to simulate what the backend does after S3 upload:
Sends a JSON webhook payload with the S3 key to the frontend.

Usage:
    python test_upload.py <frontend_webhook_url> <s3_key> [session_id] [job_type]

Example:
    python test_upload.py https://example.trycloudflare.com/api/webhook/receive/ 3d_asset_abc12345.glb abc12345 3d
"""
import sys
import json
import urllib.request


def test_s3_webhook(webhook_url, s3_key, session_id="TEST-1234", job_type="3d"):
    """Simulates the backend's webhook call after uploading a GLB to S3."""
    print(f"Sending S3 webhook notification to -> {webhook_url}")
    print(f"  S3 Key:     {s3_key}")
    print(f"  Session ID: {session_id}")
    print(f"  Job Type:   {job_type}")

    payload = json.dumps({
        "s3_key": s3_key,
        "job_type": job_type,
        "session_id": session_id
    }).encode('utf-8')

    req = urllib.request.Request(webhook_url, data=payload)
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-Session-Id', session_id)
    req.add_header('X-Job-Type', job_type)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            res_data = response.read().decode('utf-8')
            print(f"\n✅ SUCCESS! Webhook Delivery Confirmed:")
            print(f"   Django Response: {res_data}")
    except Exception as e:
        print(f"\n❌ FAILURE! Could not deliver webhook:")
        print(f"   Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_upload.py <frontend_webhook_url> <s3_key> [session_id] [job_type]")
        print("Example: python test_upload.py https://example.trycloudflare.com/api/webhook/receive/ 3d_asset_abc123.glb abc123 3d")
    else:
        url = sys.argv[1]
        key = sys.argv[2]
        sid = sys.argv[3] if len(sys.argv) > 3 else "TEST-1234"
        jtype = sys.argv[4] if len(sys.argv) > 4 else "3d"
        test_s3_webhook(url, key, sid, jtype)

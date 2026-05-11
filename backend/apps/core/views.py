from rest_framework.views import APIView
from rest_framework.response import Response


class ApiRootView(APIView):
    def get(self, request, format=None):
        return Response({
            "name": "Achernar API",
            "status": "ok",
            "endpoints": {
                "hello": "/api/hello/",
                "equipment": "/api/equipment/",
                "sessions": "/api/imaging/sessions/",
                "frames": "/api/imaging/frames/",
                "ml_models": "/api/ml/models/",
                "ml_feedback": "/api/ml/feedback/",
                "sync_register": "/api/sync/register/",
                "sync_upload_session": "/api/sync/upload-session/",
                "sync_available_models": "/api/sync/available-models/",
            },
        })


class HelloView(APIView):
    def get(self, request, format=None):
        return Response({"message": "Hello from Achernar Django API!"})

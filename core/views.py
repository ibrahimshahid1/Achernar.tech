from rest_framework.views import APIView
from rest_framework.response import Response

class HelloView(APIView):
    def get(self, request, format=None):
        return Response({"message": "Hello from Achernar Django API!"})

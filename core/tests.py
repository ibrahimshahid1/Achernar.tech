from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status


class HelloViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_json_suffix_returns_message(self):
        url = reverse('hello').rstrip('/') + '.json'
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json(), {"message": "Hello from Achernar Django API!"})

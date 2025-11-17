from django.shortcuts import render

# Create your views here.
# translator/views.py
from django.shortcuts import render

def index(request):
    return render(request, 'translator/index.html')
def about(request):
    return render(request, 'about.html')
def home(request):
    return render(request, 'home.html')
def learn_asl(request):
    return render(request, 'learn_asl.html')

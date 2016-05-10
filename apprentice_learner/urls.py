from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

from apprentice_learner import views

urlpatterns = [
	url(r'^create/$', views.create, name = "create"),
	url(r'^request/(?P<agent_id>.+)/$', views.request, name="request"),
	url(r'^train/(?P<agent_id>.+)/$', views.train, name="train"),
	url(r'^check/(?P<agent_id>.+)/$', views.check, name="check")
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

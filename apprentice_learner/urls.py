from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

from apprentice_learner import views

app_name = 'apprentice_api'
urlpatterns = [
    url(r'^create/$', views.create, name="create"),
    url(r'^request/(?P<agent_id>[0-9]+)/$', views.request, name="request"),
    url(r'^train/(?P<agent_id>[0-9]+)/$', views.train, name="train"),
    url(r'^check/(?P<agent_id>[0-9]+)/$', views.check, name="check"),
    url(r'^report/(?P<agent_id>[0-9]+)/$', views.report, name="report"),
    url(r'^request/(?P<agent_name>[a-zA-Z0-9_-]{1,200})/$',
        views.request_by_name, name="request_by_name"),
    url(r'^train/(?P<agent_name>[a-zA-Z0-9_-]{1,200})/$',
        views.train_by_name, name="train_by_name"),
    url(r'^check/(?P<agent_name>[a-zA-Z0-9_-]{1,200})/$',
        views.check_by_name, name="check_by_name"),
    url(r'^report/(?P<agent_name>[a-zA-Z0-9_-]{1,200})/$',
        views.report_by_name, name="report_by_name"),
    url(r'^tester/$', views.test_view, name='tester'),

    url(r'^get_skills/(?P<agent_id>[0-9]+)/$', views.get_skills, name='get_skills')
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from apprentice_learner import views


app_name = 'apprentice_api'
urlpatterns = [
    path('list_agents/', views.list_agents, name="list_agents"),
    path('create/', views.create, name="create"),
    path('act/', views.act, name="act"),
    path('act_all/', views.act_all, name="act_all"),
    path('act_rollout/', views.act_rollout, name="act_rollout"),
    path('train/', views.train, name="train"),
    path('train_all/', views.train_all, name="train_all"),
    path('explain_demo/', views.explain_demo, name="explain_demo"),
    path('predict_next_state/', views.predict_next_state, name="predict_next_state"),
    path('check/', views.check, name="check"),
    path('get_skills/', views.get_skills, name="get_skills"),
    
    # url(r'^report/(?P<agent_id>[0-9]+)/$', views.report, name="report"),

    # url(r'^request/(?P<agent_name>[a-zA-Z0-9_-]{1,200})/$',
    #     views.request_by_name, name="request_by_name"),
    # url(r'^train/(?P<agent_name>[a-zA-Z0-9_-]{1,200})/$',
    #     views.train_by_name, name="train_by_name"),
    # url(r'^check/(?P<agent_name>[a-zA-Z0-9_-]{1,200})/$',
    #     views.check_by_name, name="check_by_name"),
    # url(r'^report/(?P<agent_name>[a-zA-Z0-9_-]{1,200})/$',
    #     views.report_by_name, name="report_by_name"),
    # url(r'^tester/$', views.test_view, name='tester'),

    # url(r'^get_skills/(?P<agent_id>[0-9]+)/$', views.get_skills, name='get_skills')
] #+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


# DUMP
# Pattern for integers
# r'^report/(?P<agent_id>[0-9]+)/$

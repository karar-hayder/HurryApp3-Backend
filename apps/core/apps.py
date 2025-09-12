from django.apps import AppConfig

class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.core"
    def ready(self):
        
        # from .annoy_index import build_annoy_index
        # build_annoy_index()
        pass

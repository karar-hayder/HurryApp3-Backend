from django.apps import AppConfig
import sys

class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.core"

    def ready(self):
        # Avoid building index during migrations, makemigrations, or collectstatic
        if 'migrate' in sys.argv or 'makemigrations' in sys.argv or 'collectstatic' in sys.argv:
            return

        # Import and run your index builder
        from .annoy_index import build_annoy_index
        build_annoy_index()
        from .annoy_index import annoy_indexes
        # print("Annoy indexes after build:", annoy_indexes)
        # for metric, index in annoy_indexes.items():
        #     print(f"Annoy index '{metric}':", index)
        #     try:
        #         n_items = index.get_n_items() if hasattr(index, "get_n_items") else "N/A"
        #         print(f"  Number of items: {n_items}")
        #     except Exception as e:
        #         print(f"  Could not get number of items for '{metric}': {e}")

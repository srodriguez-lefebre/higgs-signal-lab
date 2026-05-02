from comet_ml import Experiment

# 1. Crear el experimento (una vez por modelo)
experiment = Experiment(
    project_name="proyecto-1-taa",       
    workspace="equipo-a"
)

# 2. Ponerle nombre y etiqueta descriptivos
experiment.set_name("DT DEFAULT")
experiment.add_tag("DT_DEFAULT")

# 3. Registrar las métricas que ya calcularon antes
experiment.log_metric("precision", precision)
experiment.log_metric("recall", recall)
experiment.log_metric("f1_score", f1)
experiment.log_metric("roc_auc", roc_auc)
experiment.log_metric("best_ams", best_ams)

# 4. Registrar el umbral óptimo de AMS como parámetro
experiment.log_parameter("best_threshold", best_threshold)

# 5. Subir la imagen de la curva AMS vs umbral
experiment.log_image("AMS_DT_DEFAULT.png")

# 6. Cerrar
experiment.end()
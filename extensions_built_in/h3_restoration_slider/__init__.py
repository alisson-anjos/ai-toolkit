from toolkit.extension import Extension

class H3RestorationSliderExtension(Extension):
    uid = 'h3_restoration_slider'
    name = 'H3 supervised restoration slider (experimental)'

    @classmethod
    def get_process(cls):
        from .trainer import H3RestorationSliderTrainer
        return H3RestorationSliderTrainer

AI_TOOLKIT_EXTENSIONS = [H3RestorationSliderExtension]

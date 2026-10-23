from toolkit.extension import Extension

class H3TemporalMultiscale(Extension):
    uid='h3_temporal_multiscale'
    name='TARA-H3 native multiscale trainer'
    @classmethod
    def get_process(cls):
        from .trainer import H3TemporalMultiscaleTrainer
        return H3TemporalMultiscaleTrainer

AI_TOOLKIT_EXTENSIONS=[H3TemporalMultiscale]

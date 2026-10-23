from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.h3_native_multiscale import validate_scales,scale_for_step

class H3TemporalMultiscaleTrainer(DiffusionTrainer):
    def __init__(self,process_id,job,config,**kwargs):
        super().__init__(process_id,job,config,**kwargs)
        kw=self.model_config.model_kwargs
        if not self.model_config.arch.startswith('minimax_h3') or not kw.get('native_temporal_alignment'):
            raise ValueError('h3_temporal_multiscale requires TARA-H3 native temporal mode')
        cfg=self.config.get('native_multiscale',{})
        self.native_scales=validate_scales(cfg.get('scales',[1.,.75,.5]))
        self.native_scale_seed=int(cfg.get('seed',42))
        kw['native_reference_training_scales']=list(self.native_scales)

    def hook_train_loop(self,batch):
        batches=batch if isinstance(batch,list) else [batch]
        scale=scale_for_step(self.native_scales,self.native_scale_seed,self.step_num)
        for micro in batches:
            micro._h3_native_reference_scale=scale
        self.additional_logs['tara/reference_scale']=scale
        if self.step_num<5 or self.step_num%50==0:
            print(f'[TARA multiscale] step={self.step_num} motion latent scale={scale}; target, image references and VLM presentation unchanged',flush=True)
        return super().hook_train_loop(batch)

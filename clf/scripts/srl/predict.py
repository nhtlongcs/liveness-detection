from src.predictor import ClsPredictor
from opt import Opts
from src.models import MODEL_REGISTRY

def predict():
    cfg = Opts().parse_args()
    resume_ckpt = cfg['global']['pretrained']
    save_path = cfg['global']['save_path'] 
    batch_sz = cfg['global']['batch_size'] 

    model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
    model = model.load_from_checkpoint(resume_ckpt, config=cfg, strict=True)
    p = ClsPredictor(model, cfg, batch_size=batch_sz)
    json_preds = p.predict_json()
    p.save(save_path, json_preds)

if __name__ == "__main__":
    predict()
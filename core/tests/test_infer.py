import pytest


def run_predict(cfg, model, mode, bs, savedir):
    from src.predictor import Predictor

    infer = Predictor(cfg=cfg, model=model, mode=mode, batch_size=bs, savedir=savedir)
    infer.predict()

@pytest.mark.skip(reason="This test will be activate when predictor is ready")
@pytest.mark.order("last")
@pytest.mark.parametrize("mode", ["simple", "complex"])
@pytest.mark.parametrize("bs", [1, 2])
def test_prediction(tmp_path, mode, bs):
    from src.models import MODEL_REGISTRY
    from opt import Opts

    resume_ckpt = "./runs/lightning_logs/version_0/checkpoints/last.ckpt"
    cfg_path = "tests/configs/inference.yml"

    cfg = Opts(cfg=cfg_path).parse_args(
        [
            "-o",
            "data.text.json_path=data_sample/meta/test_queries.json",
            "data.track.json_path=data_sample/meta/test_tracks.json",
            "data.track.image_dir=data_sample/meta/extracted_frames/",
            "data.track.motion_path=data_sample/meta/motion_map",
        ]
    )

    model = MODEL_REGISTRY.get(cfg.model["name"])(cfg)
    model.load_from_checkpoint(resume_ckpt, config=cfg)

    run_predict(cfg, model, mode, bs, tmp_path)

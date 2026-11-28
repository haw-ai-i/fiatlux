"""Psi-0 SONIC HTTP policy server with the CLIP pooled-instruction path wired in.

Runs inside the Psi0 repo's own venv (``.venv-psi``), not fiatlux's; ``scripts/psi0/serve.sh``
launches it. It is upstream's ``serve_psi0_sonic_http`` (``psi.deploy.serve_psi0_simple.Server``,
POST ``/act``) plus the one thing that server leaves out.

Why a wrapper at all: the released SONIC checkpoints are trained with
``--model.combined-temb`` -- a frozen CLIP-L pooled embedding of the instruction is added to
the diffusion timestep embedding of every action block -- so every forward pass needs
``pooled_projections``. Upstream's WebSocket server (``serve_psi0_sonic.py``) computes it with
``PooledTextEncoderCache``; upstream's HTTP server never does, and the action head then fails
inside ``CombinedTimestepTextProjEmbeddings`` (``linear(): argument 'input' must be Tensor, not
NoneType``) on the first request. The WebSocket server cannot stand in: it runs its own
wall-clock 30 Hz control loop, while the benchmark pauses the sim clock for every request.
So this reuses upstream's own ``PooledTextEncoderCache`` (training-time cache first, frozen
``openai/clip-vit-large-patch14`` for an unseen instruction -- exactly the WebSocket server's
behavior) and hands its output to the model's ``pooled_projections`` argument, which all three
of ``predict_action`` / ``predict_action_with_rtc_flow`` / ``predict_action_with_training_rtc_flow``
already accept. Nothing else about the HTTP server changes.
"""

import sys
from pathlib import Path

import tyro
from psi.config.config import ServerConfig
from psi.deploy.serve_psi0_simple import Server
from psi.deploy.serve_psi0_sonic import PooledTextEncoderCache
from psi.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

_POOLED_METHODS = ("predict_action", "predict_action_with_rtc_flow", "predict_action_with_training_rtc_flow")


class PooledServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mc = self.launch_config.model
        self.pooled_helper = None
        if getattr(mc, "pooled_text_encoder", None) is None:
            return
        cache_rel = getattr(mc, "pooled_cache_path", None)
        cache_file = None
        if cache_rel is not None:
            cache_file = cache_rel if Path(cache_rel).is_absolute() else str(Path(self.run_dir) / cache_rel)
        self.pooled_helper = PooledTextEncoderCache(
            cache_path=cache_file,
            encoder=mc.pooled_text_encoder,
            encoder_path=mc.pooled_text_encoder_path,
            projection_dim=mc.pooled_projection_dim,
            device=self.device,
        )
        for name in _POOLED_METHODS:
            self._inject_pooled(name)
        overwatch.info(f"combined_temb checkpoint: pooled {mc.pooled_text_encoder} projections wired into /act")

    def _inject_pooled(self, name: str) -> None:
        original = getattr(self.model, name)

        def with_pooled(*args, **kwargs):
            if kwargs.get("pooled_projections") is None:
                instructions = kwargs["instructions"]
                assert len(instructions) == 1, "the HTTP server is single-sample (B=1)"
                kwargs["pooled_projections"] = self.pooled_helper(instructions[0])
            return original(*args, **kwargs)

        setattr(self.model, name, with_pooled)


def main():
    from dotenv import load_dotenv

    load_dotenv()
    cfg = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[1:])
    assert cfg.policy is not None, "which policy to serve?"
    server = PooledServer(
        cfg.policy,
        Path(cfg.run_dir),
        cfg.ckpt_step,
        cfg.device,
        cfg.rtc,
        cfg.action_exec_horizon,
        rtc_mode=cfg.rtc_mode,
        pig_mask_schedule=cfg.pig_mask_schedule,
        pig_guidance_alpha=cfg.pig_guidance_alpha,
        num_inference_steps=cfg.num_inference_steps,
        rtc_inference_delay=cfg.rtc_inference_delay,
        min_exec_horizon=cfg.min_exec_horizon,
    )
    server.run(cfg.host, cfg.port)


if __name__ == "__main__":
    main()

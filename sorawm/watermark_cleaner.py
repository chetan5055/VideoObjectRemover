from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, E2FGVIHDConfig
from sorawm.cleaner.lama_cleaner import LamaCleaner
from sorawm.schemas import CleanerType


class WaterMarkCleaner:
    def __new__(
        cls,
        cleaner_type: CleanerType,
        enable_torch_compile: bool,
        use_bf16: bool = False,
        # ✅ add these (safe defaults)
        propainter_dir: str | None = None,
        propainter_weights_dir: str | None = None,
        device: str = "cuda",
        propainter_fast_mode: bool = True,
    ):
        if cleaner_type == CleanerType.LAMA:
            return LamaCleaner()

        if cleaner_type == CleanerType.E2FGVI_HQ:
            e2fgvi_hq_config = E2FGVIHDConfig(
                enable_torch_compile=enable_torch_compile,
                use_bf16=use_bf16,
            )
            return E2FGVIHDCleaner(config=e2fgvi_hq_config)

        if cleaner_type == CleanerType.PROPAINTER:
            if not propainter_dir:
                raise ValueError("propainter_dir is required for PROPAINTER cleaner")
            from sorawm.cleaner.propainter_cleaner import ProPainterCleaner
            return ProPainterCleaner(
                propainter_dir=propainter_dir,
                weights_dir=propainter_weights_dir,
                device=device,
                fast_mode=propainter_fast_mode,
            )

        raise ValueError(f"Invalid cleaner type: {cleaner_type}")

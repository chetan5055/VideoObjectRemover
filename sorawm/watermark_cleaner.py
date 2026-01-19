from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, E2FGVIHDConfig
from sorawm.cleaner.lama_cleaner import LamaCleaner
from sorawm.schemas import CleanerType


class WaterMarkCleaner:
    def __new__(
        cls,
        cleaner_type: CleanerType,
        enable_torch_compile: bool,
        use_bf16: bool = False,
    ):
        """
        Factory that returns an instance of the requested cleaner.
        """

        if cleaner_type == CleanerType.LAMA:
            return LamaCleaner()

        elif cleaner_type == CleanerType.E2FGVI_HQ:
            e2fgvi_hq_config = E2FGVIHDConfig(
                enable_torch_compile=enable_torch_compile,
                use_bf16=use_bf16,
            )
            return E2FGVIHDCleaner(config=e2fgvi_hq_config)

        elif cleaner_type == CleanerType.PROPAINTER:
            # ProPainterCleaner must exist and match the interface used by core.py
            from sorawm.cleaner.propainter_cleaner import ProPainterCleaner
            return ProPainterCleaner()

        else:
            raise ValueError(f"Invalid cleaner type: {cleaner_type}")

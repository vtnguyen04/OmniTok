"""Global registry system for models, losses, teachers, and trainers.

Inspired by RADIO's AdaptorRegistry pattern. Provides a unified way to
register and instantiate components by name, eliminating if-else chains
and enabling config-driven construction.

This module provides:
    - Registry: Generic registry class
    - Pre-instantiated global registries for each component type
"""

from typing import Any, Callable, Dict, List, Optional, Type


class Registry:
    """A generic registry that maps string names to classes or factory functions.

    Supports decorator-based registration and config-driven instantiation.
    Designed to be used as a singleton per component type.

    Args:
        name: Human-readable name for this registry (e.g., "Encoder", "Loss").

    Example:
        >>> ENCODER_REGISTRY = Registry("Encoder")
        >>> @ENCODER_REGISTRY.register("vit_large")
        ... class ViTLargeEncoder:
        ...     pass
        >>> encoder = ENCODER_REGISTRY.build("vit_large", embed_dim=1024)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str) -> Callable:
        """Register a class or factory function under the given name.

        Args:
            name: Unique identifier for this component.

        Returns:
            Decorator that registers the class.

        Raises:
            ValueError: If name is already registered.
        """

        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(f"[{self._name}] '{name}' is already registered. Available: {self.available()}")
            self._registry[name] = cls
            return cls

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered component by name.

        Args:
            name: Registered component name.
            **kwargs: Arguments passed to the component constructor.

        Returns:
            Instantiated component.

        Raises:
            KeyError: If name is not registered.
        """
        if name not in self._registry:
            raise KeyError(f"[{self._name}] '{name}' is not registered. Available: {self.available()}")
        return self._registry[name](**kwargs)

    def get(self, name: str) -> Optional[Type]:
        """Get the registered class without instantiating.

        Args:
            name: Registered component name.

        Returns:
            The registered class, or None if not found.
        """
        return self._registry.get(name)

    def available(self) -> List[str]:
        """List all registered component names.

        Returns:
            Sorted list of registered names.
        """
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name='{self._name}', components={self.available()})"


# ---------------------------------------------------------------------------
# Global registries — import these wherever you need to register or build
# ---------------------------------------------------------------------------

ENCODER_REGISTRY = Registry("Encoder")
DECODER_REGISTRY = Registry("Decoder")
BOTTLENECK_REGISTRY = Registry("Bottleneck")
HEAD_REGISTRY = Registry("Head")
PROJECTOR_REGISTRY = Registry("Projector")
DIT_REGISTRY = Registry("DiT")
TEACHER_REGISTRY = Registry("Teacher")
ALIGNMENT_REGISTRY = Registry("Alignment")
LOSS_REGISTRY = Registry("Loss")
TEACHER_ROUTER_REGISTRY = Registry("TeacherRouter")
TRAINER_REGISTRY = Registry("Trainer")

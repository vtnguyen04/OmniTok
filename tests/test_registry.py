"""Tests for the Registry pattern."""

import pytest

from omnitok.registry import (
    ALIGNMENT_REGISTRY,
    ENCODER_REGISTRY,
    TEACHER_REGISTRY,
    Registry,
)


class TestRegistry:
    """Unit tests for Registry class."""

    def test_register_and_build(self) -> None:
        """Registered class can be built by name."""
        reg = Registry("Test")

        @reg.register("dummy")
        class DummyClass:
            def __init__(self, value: int = 42) -> None:
                self.value = value

        obj = reg.build("dummy", value=99)
        assert obj.value == 99

    def test_register_duplicate_raises(self) -> None:
        """Registering the same name twice raises ValueError."""
        reg = Registry("Test")

        @reg.register("foo")
        class Foo:
            pass

        with pytest.raises(ValueError, match="already registered"):
            @reg.register("foo")
            class FooAgain:
                pass

    def test_build_unregistered_raises(self) -> None:
        """Building an unregistered name raises KeyError."""
        reg = Registry("Test")

        with pytest.raises(KeyError, match="not registered"):
            reg.build("nonexistent")

    def test_available_returns_sorted(self) -> None:
        """available() returns sorted list of registered names."""
        reg = Registry("Test")

        @reg.register("beta")
        class Beta:
            pass

        @reg.register("alpha")
        class Alpha:
            pass

        assert reg.available() == ["alpha", "beta"]

    def test_contains(self) -> None:
        """__contains__ works for membership check."""
        reg = Registry("Test")

        @reg.register("exists")
        class Exists:
            pass

        assert "exists" in reg
        assert "missing" not in reg

    def test_len(self) -> None:
        """__len__ returns number of registered components."""
        reg = Registry("Test")
        assert len(reg) == 0

        @reg.register("one")
        class One:
            pass

        assert len(reg) == 1

    def test_get_returns_class(self) -> None:
        """get() returns the class without instantiating."""
        reg = Registry("Test")

        @reg.register("cls")
        class MyClass:
            pass

        assert reg.get("cls") is MyClass
        assert reg.get("missing") is None

    def test_repr(self) -> None:
        """__repr__ shows registry name and components."""
        reg = Registry("Encoder")

        @reg.register("vit")
        class ViT:
            pass

        result = repr(reg)
        assert "Encoder" in result
        assert "vit" in result

    def test_build_with_no_args(self) -> None:
        """Build works with default constructor arguments."""
        reg = Registry("Test")

        @reg.register("simple")
        class Simple:
            def __init__(self) -> None:
                self.ready = True

        obj = reg.build("simple")
        assert obj.ready is True


class TestGlobalRegistries:
    """Verify global registries are independent instances."""

    def test_registries_are_independent(self) -> None:
        """Global registries don't share state."""
        assert ENCODER_REGISTRY._name == "Encoder"
        assert TEACHER_REGISTRY._name == "Teacher"
        assert ALIGNMENT_REGISTRY._name == "Alignment"
        assert ENCODER_REGISTRY is not TEACHER_REGISTRY
        assert ENCODER_REGISTRY is not ALIGNMENT_REGISTRY

    def test_registries_are_registry_instances(self) -> None:
        """Global registries are Registry instances."""
        assert isinstance(ENCODER_REGISTRY, Registry)
        assert isinstance(TEACHER_REGISTRY, Registry)
        assert isinstance(ALIGNMENT_REGISTRY, Registry)

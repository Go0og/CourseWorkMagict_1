"""Relation resolution stage (stub for v0.1).

Line/connection detection is out of scope for the MVP: this resolver
always returns an empty relation list so classes can be validated first.
"""

from __future__ import annotations

from uml_recognition.models import Relation, UmlClass


class RelationResolver:
    """Matches detected lines between class boxes into UML relations."""

    def resolve(self, classes: list[UmlClass]) -> list[Relation]:
        """Resolve relations between the given classes.

        Args:
            classes: Detected UML classes (unused in the stub).

        Returns:
            An empty list - relation detection is not implemented in v0.1.
        """
        print(f"[RelationResolver] Stub: 0 relations for {len(classes)} class(es)")
        return []

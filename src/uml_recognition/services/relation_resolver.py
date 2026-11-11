"""Relation resolution stage (stub for iteration 3).

Detecting UML edges (association / inheritance lines) between classes
will be implemented in a later iteration. For now this returns an
empty list so the pipeline contract stays stable.
"""

from __future__ import annotations

from uml_recognition.models import Relation, UmlClass


class RelationResolver:
    """Resolves relations between detected UML classes.

    STUB: always returns an empty list of relations.
    """

    def resolve(self, classes: list[UmlClass], lines: list) -> list[Relation]:
        """Find relations between the given classes.

        Args:
            classes: Detected UML classes.
            lines: Detected connector lines (unused in the stub).

        Returns:
            An empty list — relation detection is not implemented yet.
        """
        print(f"[RelationResolver] Stub: {len(classes)} classes -> 0 relations")
        return []

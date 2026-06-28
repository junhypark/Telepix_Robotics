"""External inspection module separated from vision and control."""

from __future__ import annotations

import time

from robot_sorting.schemas import DetectedObject, ExternalInspectionResult


class InspectionModule:
    """Convert detected objects into inspection results."""

    def inspect(self, detected_object: DetectedObject) -> ExternalInspectionResult:
        """Inspect one detected object and preserve the external decision boundary."""

        return ExternalInspectionResult(
            object_id=detected_object.object_id,
            label=detected_object.label,
            world_position=detected_object.world_position,
            confidence=detected_object.confidence,
            inspected_at=time.perf_counter(),
        )

    def inspect_all(self, detected_objects: list[DetectedObject]) -> list[ExternalInspectionResult]:
        """Inspect all detections in deterministic order."""

        return [self.inspect(obj) for obj in detected_objects]


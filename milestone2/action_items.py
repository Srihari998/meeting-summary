"""
Action Items Module
Provides prompt-side specifications and post-processing normalizations for
meeting action items:
- task: clear action description
- assignee: person or team responsible
- deadline: target date or None
- priority: High / Medium / Low (defaults to Medium if not specified)
- status: always starts as "Not Started"
"""

from typing import List
try:
    from schemas import ActionItem
except ImportError:
    from .schemas import ActionItem


DEFAULT_PRIORITY = "Medium"
DEFAULT_STATUS = "Not Started"
VALID_PRIORITIES = {"High", "Medium", "Low"}


def normalize_action_item(item: ActionItem) -> ActionItem:
    """
    Ensures an action item has standardized priority and status.
    - If priority is not valid or missing, defaults to 'Medium'.
    - Status is standardized to start at 'Not Started' unless explicitly in progress.
    """
    priority = item.priority if item.priority in VALID_PRIORITIES else DEFAULT_PRIORITY
    status = item.status.strip() if item.status and item.status.strip() else DEFAULT_STATUS
    assignee = item.assignee.strip() if item.assignee and item.assignee.strip() else "Unassigned"

    return ActionItem(
        task=item.task.strip(),
        assignee=assignee,
        deadline=item.deadline.strip() if item.deadline else None,
        priority=priority,
        status=status,
    )


def normalize_action_items(items: List[ActionItem]) -> List[ActionItem]:
    """Applies normalization across a list of action items."""
    return [normalize_action_item(it) for it in items]
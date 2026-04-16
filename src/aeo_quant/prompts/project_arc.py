"""Project Arc prompt bank -- multi-turn coding task queue for context scaling tests.

Provides a system message, initial spec, and progressive difficulty bands that
exercise increasing context windows. The ``select_prompt`` function picks the
right band based on current fill ratio.

Stdlib only -- no external dependencies.
"""
from __future__ import annotations

SYSTEM_MESSAGE = (
    "You are a senior Python engineer. The user will ask you to iteratively "
    "build and improve a priority task queue system. Write production-quality "
    "code and explain your design decisions. When asked to review or refactor, "
    "reference specific parts of your earlier code in this conversation."
)

INITIAL_SPEC = """\
Build a priority task queue system in Python with the following requirements:

1. **Task model**: Each task has an id (UUID), a priority (1=highest, 5=lowest), \
a payload (arbitrary dict), a status (pending/running/completed/failed/cancelled), \
timestamps (created_at, started_at, completed_at), and an optional result field.

2. **PriorityQueue class**:
   - `submit(payload, priority=3) -> Task` -- enqueue a new task
   - `next() -> Task | None` -- dequeue highest-priority task (FIFO within same priority)
   - `complete(task_id, result)` -- mark a task as completed with its result
   - `fail(task_id, error)` -- mark a task as failed with error info
   - `cancel(task_id)` -- cancel a pending task (no-op if already running/done)
   - `status(task_id) -> Task` -- get current task state
   - `stats() -> dict` -- return counts by status and by priority

3. **Worker pool**: A `WorkerPool` class that:
   - Takes a queue and a number of workers
   - Each worker runs in a thread, pulling tasks via `next()`
   - Workers call a user-supplied `handler(task) -> result` function
   - Graceful shutdown via `pool.shutdown(timeout=30)`

4. **Storage**: Start with in-memory storage, but design the interface so it \
could be swapped for Redis/SQLite later.

Here's the skeleton to start from:

```python
import uuid
import heapq
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 3
    payload: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: Any = None

# TODO: Implement PriorityQueue
# TODO: Implement WorkerPool
# TODO: Add proper error handling
# TODO: Add graceful shutdown
```

Implement the full system. Make sure the priority queue correctly handles FIFO \
ordering within the same priority level."""

PROMPTS_LIGHT = [
    ("timeout", "Add a `task_timeout` parameter that automatically fails tasks exceeding their deadline."),
    ("inspect", "Add a method to inspect queue state -- pending count by priority, active workers, cancelled tasks."),
    ("dependencies", "Add task dependencies -- a task can declare prerequisite task IDs that must complete first."),
    ("retries", "Add a `max_retries` parameter with exponential backoff for failed tasks."),
]

PROMPTS_MEDIUM = [
    ("cancel_tests", "Write comprehensive unit tests for the cancellation logic. Cover: cancel pending, cancel running, cancel completed, cancel nonexistent."),
    ("concurrency", "Your implementation isn't safe under concurrent access. Identify the race conditions and fix them with proper synchronization."),
    ("logging", "Add structured logging with correlation IDs so you can trace a task's full lifecycle from submission to completion."),
    ("load_test", "Write integration tests that simulate 100 concurrent task submissions with mixed priorities."),
]

PROMPTS_HEAVY = [
    ("crash_recovery", "What happens if a worker crashes mid-task with the lock held? Design and implement a recovery mechanism with lease-based expiry."),
    ("complexity", "Review the full implementation across this conversation. Identify the three operations with worst time complexity and optimize them."),
    ("throughput", "The system needs to handle 50,000 tasks/second. What are the bottlenecks in your current design? Restructure the core data structures."),
    ("persistence", "Implement a persistence layer so the queue survives process restarts. Design the WAL format and recovery procedure."),
    ("distributed", "Add distributed coordination -- multiple queue instances sharing state. Handle split-brain scenarios."),
    ("memory_profile", "Analyze the memory profile of your implementation. Where does memory grow unbounded? Add backpressure."),
]

PROMPTS_TAIL = [
    ("fragile", "Look at everything you've written in this conversation. What's the most fragile part? Harden it."),
    ("leak", "Find the biggest abstraction leak in your current design and fix it."),
    ("invariants", "What invariants should hold across your entire implementation? Write assertions that verify them."),
    ("refactor", "Pick the most complex function you've written and refactor it for clarity without changing behavior."),
]


def select_prompt(
    turn: int, fill_ratio: float, band_counters: dict[str, int],
) -> tuple[str, str, str]:
    """Return (label, text, difficulty) for the given turn and fill ratio.

    Turn 0 = initial spec.  Turns 1+ draw sequentially from the prompt
    bank, advancing through difficulty bands as fill_ratio grows.
    If a band is exhausted before fill_ratio advances, cycle tail prompts.

    band_counters is mutated to track per-band usage across turns.
    """
    if turn == 0:
        return "initial_spec", INITIAL_SPEC, "light"

    # Determine which band we're in based on fill ratio
    if fill_ratio < 0.25:
        band, prompts = "light", PROMPTS_LIGHT
    elif fill_ratio < 0.50:
        band, prompts = "medium", PROMPTS_MEDIUM
    elif fill_ratio < 0.80:
        band, prompts = "heavy", PROMPTS_HEAVY
    else:
        band, prompts = "tail", PROMPTS_TAIL

    idx = band_counters.get(band, 0)

    if idx < len(prompts):
        label, text = prompts[idx]
        band_counters[band] = idx + 1
        return label, text, band

    # Band exhausted -- cycle tail prompts
    tail_idx = band_counters.get("_tail_overflow", 0)
    label, text = PROMPTS_TAIL[tail_idx % len(PROMPTS_TAIL)]
    band_counters["_tail_overflow"] = tail_idx + 1
    return label, text, "tail"

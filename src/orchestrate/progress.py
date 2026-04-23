from __future__ import annotations

import os
import socket
import sys
import threading
import time
from dataclasses import dataclass


@dataclass(slots=True)
class FileProgressState:
    index: int
    name: str
    status: str = "queued"
    lease_status: str = "queued"
    lease_owner: str = ""
    stage: str = "queued"
    detail: str = ""
    candidates: int = 0
    accepted: int = 0
    unresolved: int = 0
    priority_boost: bool = False
    cancel_requested: bool = False
    started_at: float | None = None
    finished_at: float | None = None


@dataclass(slots=True)
class PipelineProgressSnapshot:
    total_files: int
    queued: int
    active: int
    completed: int
    skipped: int
    cancelled: int
    unsupported: int
    extracted: int
    unresolved: int
    elapsed_seconds: float


class PipelineProgressReporter:
    def __init__(self, total_files: int, mode: str = "auto", enabled: bool = True) -> None:
        self.total_files = total_files
        self.enabled = enabled and total_files >= 0
        self.started_at = time.perf_counter()
        self._lock = threading.RLock()
        self._states: dict[int, FileProgressState] = {}
        self._extracted = 0
        self._unresolved = 0
        self._unsupported = 0
        self._recent_logs: list[str] = []
        self._mode = self._resolve_mode(mode) if self.enabled else "plain"
        self._screen = None
        self._curses = None
        self._supports_color = False
        self._paused = False
        self._stop_requested = False
        self._abort_requested = False
        self._scroll_offset = 0
        self._selected_row = 0
        self._filter_mode = "all"
        self._sort_mode = "index"
        self._show_inspector = False
        self._last_render_at = 0.0
        self._last_tick_at = 0.0
        self._local_owner = socket.gethostname()
        if self.enabled and self._mode == "ncurses":
            self._init_curses()

    def close(self) -> None:
        with self._lock:
            if self._mode == "ncurses" and self._curses is not None and self._screen is not None:
                try:
                    self._curses.nocbreak()
                    self._screen.keypad(False)
                    self._curses.echo()
                    self._curses.endwin()
                except Exception:
                    pass
                finally:
                    self._screen = None
                    self._curses = None

    def queue_file(self, index: int, name: str) -> None:
        with self._lock:
            self._states[index] = FileProgressState(index=index, name=name)
            self._render()

    def sync_shared_state(
        self,
        index: int,
        name: str,
        *,
        status: str,
        lease_status: str | None = None,
        lease_owner: str = "",
        stage: str,
        detail: str = "",
        started_at: float | None = None,
        finished_at: float | None = None,
        candidates: int | None = None,
        accepted: int | None = None,
        unresolved: int | None = None,
    ) -> None:
        normalized_status = {
            "completed": "done",
            "failed": "failed",
        }.get(status, status)
        with self._lock:
            state = self._states.setdefault(index, FileProgressState(index=index, name=name))
            state.status = normalized_status
            state.lease_status = lease_status or status
            state.lease_owner = lease_owner
            state.stage = stage or normalized_status
            state.detail = detail
            if started_at is not None:
                state.started_at = started_at
            if finished_at is not None:
                state.finished_at = finished_at
            elif normalized_status not in {"running", "queued"}:
                state.finished_at = time.perf_counter()
            if candidates is not None:
                state.candidates = candidates
            if accepted is not None:
                state.accepted = accepted
            if unresolved is not None:
                state.unresolved = unresolved
            self._render()

    def skip_file(self, index: int, name: str, reason: str) -> None:
        with self._lock:
            state = self._states.setdefault(index, FileProgressState(index=index, name=name))
            state.status = "skipped"
            state.lease_status = "skipped"
            state.stage = "skipped"
            state.detail = reason
            state.finished_at = time.perf_counter()
            self._log_line(f"{self._prefix()} {name} :: skipped - {reason}")
            self._render()

    def cancel_file(self, index: int, reason: str = "cancelled by user") -> bool:
        with self._lock:
            state = self._states.get(index)
            if state is None:
                return False
            if state.status == "queued":
                state.status = "cancelled"
                state.lease_status = "cancelled"
                state.stage = "cancelled"
                state.detail = reason
                state.finished_at = time.perf_counter()
                self._log_line(f"{self._prefix()} {state.name} :: cancelled - {reason}")
                self._render()
                return True
            if state.status == "running":
                state.cancel_requested = True
                state.detail = f"{state.detail} | cancel requested".strip(" |")
                self._log_line(f"{self._prefix()} {state.name} :: cancel requested for running task")
                self._render()
                return False
            return False

    def prioritize_file(self, index: int) -> bool:
        with self._lock:
            state = self._states.get(index)
            if state is None or state.status != "queued":
                return False
            for other in self._states.values():
                other.priority_boost = False
            state.priority_boost = True
            state.detail = "priority dispatch requested"
            self._log_line(f"{self._prefix()} {state.name} :: prioritized for next dispatch")
            self._render()
            return True

    def mark_unsupported(self, index: int, name: str, reason: str) -> None:
        with self._lock:
            state = self._states.setdefault(index, FileProgressState(index=index, name=name))
            state.status = "unsupported"
            state.lease_status = "unsupported"
            state.stage = "unsupported"
            state.detail = reason
            state.finished_at = time.perf_counter()
            self._unsupported += 1
            self._unresolved += 1
            self._log_line(f"{self._prefix()} {name} :: unsupported - {reason}")
            self._render()

    def start_file(self, index: int, name: str) -> None:
        with self._lock:
            state = self._states.setdefault(index, FileProgressState(index=index, name=name))
            state.status = "running"
            state.lease_status = "running"
            state.lease_owner = self._local_owner
            state.stage = "starting"
            state.detail = ""
            state.started_at = time.perf_counter()
            self._log_line(f"{self._prefix()} {name} :: started")
            self._render()

    def update_stage(self, index: int, name: str, stage: str, detail: str = "") -> None:
        with self._lock:
            state = self._states.setdefault(index, FileProgressState(index=index, name=name))
            state.status = "running"
            state.lease_status = "running"
            state.lease_owner = self._local_owner
            state.stage = stage
            state.detail = detail
            self._log_line(f"{self._prefix()} {name} :: {stage}{' - ' + detail if detail else ''}")
            self._render()

    def update_candidates(self, index: int, name: str, candidates: int) -> None:
        with self._lock:
            state = self._states.setdefault(index, FileProgressState(index=index, name=name))
            state.candidates = candidates
            self._log_line(f"{self._prefix()} {name} :: extracted {candidates} candidate point(s)")
            self._render()

    def record_unresolved(self, index: int, name: str, count: int = 1, detail: str = "") -> None:
        with self._lock:
            state = self._states.setdefault(index, FileProgressState(index=index, name=name))
            state.unresolved += count
            self._unresolved += count
            self._log_line(f"{self._prefix()} {name} :: unresolved +{count}{' - ' + detail if detail else ''}")
            self._render()

    def complete_file(self, index: int, name: str, accepted: int) -> None:
        with self._lock:
            state = self._states.setdefault(index, FileProgressState(index=index, name=name))
            state.status = "done"
            state.lease_status = "completed"
            state.lease_owner = self._local_owner
            state.stage = "done"
            state.accepted = accepted
            state.finished_at = time.perf_counter()
            self._extracted += accepted
            elapsed = self._state_elapsed(state)
            self._log_line(f"{self._prefix()} {name} :: done - accepted {accepted} point(s) in {elapsed}")
            self._render()

    def emit_global(self, message: str) -> None:
        with self._lock:
            self._log_line(message)
            self._render()

    def tick(self) -> None:
        with self._lock:
            self._poll_input()
            now = time.perf_counter()
            if now - self._last_tick_at >= 0.25:
                self._last_tick_at = now
                self._render(force=True)

    def pause_requested(self) -> bool:
        with self._lock:
            self._poll_input()
            return self._paused

    def stop_requested(self) -> bool:
        with self._lock:
            self._poll_input()
            return self._stop_requested

    def abort_requested(self) -> bool:
        with self._lock:
            self._poll_input()
            return self._abort_requested

    def selected_index(self) -> int | None:
        with self._lock:
            visible = self._visible_states()
            if not visible:
                return None
            self._selected_row = max(0, min(len(visible) - 1, self._selected_row))
            return visible[self._selected_row].index

    def pick_pending_index(self, pending_indexes: list[int]) -> int | None:
        with self._lock:
            pending_set = set(pending_indexes)
            boosted = [
                state.index
                for state in self._states.values()
                if state.index in pending_set and state.status == "queued" and state.priority_boost and not state.cancel_requested
            ]
            if boosted:
                return boosted[0]
            available = [
                state.index
                for state in self._states.values()
                if state.index in pending_set and state.status == "queued" and not state.cancel_requested
            ]
            return min(available) if available else None

    def note_dispatch_started(self, index: int) -> None:
        with self._lock:
            state = self._states.get(index)
            if state is not None:
                state.priority_boost = False

    def is_cancelled_before_dispatch(self, index: int) -> bool:
        with self._lock:
            state = self._states.get(index)
            return state is not None and state.status == "cancelled"

    def status_for(self, index: int) -> str | None:
        with self._lock:
            state = self._states.get(index)
            return state.status if state is not None else None

    def snapshot(self) -> PipelineProgressSnapshot:
        queued = sum(1 for state in self._states.values() if state.status == "queued")
        active = sum(1 for state in self._states.values() if state.status == "running")
        completed = sum(1 for state in self._states.values() if state.status == "done")
        skipped = sum(1 for state in self._states.values() if state.status == "skipped")
        cancelled = sum(1 for state in self._states.values() if state.status == "cancelled")
        unsupported = sum(1 for state in self._states.values() if state.status == "unsupported")
        extracted = sum(state.accepted for state in self._states.values())
        unresolved = sum(state.unresolved for state in self._states.values())
        elapsed = time.perf_counter() - self.started_at
        return PipelineProgressSnapshot(
            total_files=self.total_files,
            queued=queued,
            active=active,
            completed=completed,
            skipped=skipped,
            cancelled=cancelled,
            unsupported=unsupported,
            extracted=extracted,
            unresolved=unresolved,
            elapsed_seconds=elapsed,
        )

    def _resolve_mode(self, mode: str) -> str:
        requested = (mode or "auto").lower()
        if requested == "plain":
            return "plain"
        if requested == "ncurses":
            return "ncurses" if self._can_use_ncurses() else "plain"
        return "ncurses" if self._can_use_ncurses() else "plain"

    def _can_use_ncurses(self) -> bool:
        if not sys.stdout.isatty():
            return False
        term = os.environ.get("TERM", "")
        if not term or term == "dumb":
            return False
        try:
            import curses  # noqa: F401
        except ImportError:
            return False
        return True

    def _init_curses(self) -> None:
        import curses

        self._curses = curses
        self._screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self._screen.keypad(True)
        self._screen.nodelay(True)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
            except curses.error:
                pass
            self._init_color_pairs()

    def _init_color_pairs(self) -> None:
        if self._curses is None:
            return
        curses = self._curses
        self._supports_color = True
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_BLUE, -1)
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)
        curses.init_pair(7, curses.COLOR_WHITE, -1)

    def _render(self, force: bool = False) -> None:
        if not self.enabled:
            return
        if self._mode == "ncurses":
            now = time.perf_counter()
            if force or now - self._last_render_at >= 0.08:
                self._last_render_at = now
                self._render_ncurses()

    def _render_ncurses(self) -> None:
        if self._screen is None or self._curses is None:
            return
        screen = self._screen
        curses = self._curses
        snapshot = self.snapshot()
        self._poll_input()
        screen.erase()
        height, width = screen.getmaxyx()

        runtime = _format_elapsed(snapshot.elapsed_seconds)
        clock = time.strftime("%H:%M:%S")
        mode = "PAUSED" if self._paused else "RUNNING"
        if self._stop_requested:
            mode = "STOPPING"
        if self._abort_requested:
            mode = "ABORTING"
        header = (
            f"LITTORAL  {mode}  clock {clock}  runtime {runtime}  "
            f"done {snapshot.completed + snapshot.skipped + snapshot.unsupported + snapshot.cancelled}/{snapshot.total_files}  "
            f"active {snapshot.active}  queued {snapshot.queued}  extracted {snapshot.extracted}  unresolved {snapshot.unresolved}"
        )
        self._add_line(0, 0, header, width - 1, self._style_for_mode(mode) | curses.A_BOLD)
        self._add_line(1, 0, "=" * max(1, width - 1), width - 1, self._style("dim"))
        controls = "Controls: p pause  s stop  q abort-queue  t trigger  x cancel  i inspect  r refresh"
        self._add_line(2, 0, controls, width - 1, self._style("accent"))
        browser = (
            f"Browse: j/k or arrows move  PgUp/PgDn page  g/G top/bottom  "
            f"f filter={self._filter_mode}  o sort={self._sort_mode}"
        )
        self._add_line(3, 0, browser, width - 1, self._style("accent"))
        self._add_line(4, 0, "Idx Name                    Status   Lease    Owner          Stage        Time  Cnd Acc Unr Detail", width - 1, self._style("header"))
        self._add_line(5, 0, "-" * max(1, width - 1), width - 1, self._style("dim"))

        visible_states = self._visible_states()
        table_rows = max(4, height - 13)
        self._clamp_viewport(len(visible_states), table_rows)
        row = 6
        window = visible_states[self._scroll_offset : self._scroll_offset + table_rows]
        for absolute_index, state in enumerate(window, start=self._scroll_offset):
            runtime_text = self._state_elapsed(state)
            line = (
                f"{state.index:>3}. "
                f"{state.name[:23]:23} "
                f"{state.status[:8]:8} "
                f"{state.lease_status[:8]:8} "
                f"{state.lease_owner[:14]:14} "
                f"{state.stage[:12]:12} "
                f"{runtime_text:>10} "
                f"{state.candidates:>3} "
                f"{state.accepted:>3} "
                f"{state.unresolved:>3} "
                f"{state.detail}"
            )
            style = self._style_for_state(state)
            if absolute_index == self._selected_row:
                style |= curses.A_REVERSE
            self._add_line(row, 0, line, width - 1, style)
            row += 1

        footer = (
            f"Rows {self._scroll_offset + 1 if visible_states else 0}-{min(self._scroll_offset + table_rows, len(visible_states))}"
            f"/{len(visible_states)} visible"
        )
        self._add_line(row, 0, footer, width - 1, self._style("dim"))

        log_divider = min(height - 6, row + 1)
        self._add_line(log_divider, 0, "=" * max(1, width - 1), width - 1, self._style("dim"))
        if self._show_inspector:
            self._render_inspector(log_divider + 1, height, width)
        else:
            self._add_line(log_divider + 1, 0, "Recent Events", width - 1, self._style("header"))
            for offset, line in enumerate(self._recent_logs[-max(1, height - log_divider - 3):], start=0):
                target_row = log_divider + 2 + offset
                if target_row >= height:
                    break
                self._add_line(target_row, 0, line, width - 1, self._style("plain"))

        screen.refresh()

    def _poll_input(self) -> None:
        if self._mode != "ncurses" or self._screen is None:
            return
        while True:
            try:
                code = self._screen.getch()
            except Exception:
                return
            if code == -1:
                return
            raw_key = chr(code) if 0 <= code < 256 else ""
            key = raw_key.lower()
            if key == "p":
                self._paused = not self._paused
                self._log_line(self._timestamped(f"control :: {'paused' if self._paused else 'resumed'} dispatch"))
            elif key == "s":
                self._stop_requested = True
                self._paused = False
                self._log_line(self._timestamped("control :: graceful stop requested"))
            elif key == "q":
                self._abort_requested = True
                self._stop_requested = True
                self._paused = False
                self._log_line(self._timestamped("control :: abort requested; queued work will stop dispatching"))
            elif key == "r":
                self._log_line(self._timestamped("control :: manual refresh"))
            elif key == "t":
                self._trigger_selected()
            elif key == "x":
                self._cancel_selected()
            elif key == "i":
                self._show_inspector = not self._show_inspector
            elif key in {"j"} or code == getattr(self._curses, "KEY_DOWN", -9999):
                self._move_selection(1)
            elif key in {"k"} or code == getattr(self._curses, "KEY_UP", -9999):
                self._move_selection(-1)
            elif code == getattr(self._curses, "KEY_NPAGE", -9999):
                self._page_selection(1)
            elif code == getattr(self._curses, "KEY_PPAGE", -9999):
                self._page_selection(-1)
            elif raw_key == "g":
                self._selected_row = 0
                self._scroll_offset = 0
            elif raw_key == "G":
                self._jump_bottom()
            elif key == "f":
                self._cycle_filter()
            elif key == "o":
                self._cycle_sort()
            elif key == "/":
                self._cycle_filter()
            self._render(force=True)

    def _visible_states(self) -> list[FileProgressState]:
        states = list(self._states.values())
        if self._filter_mode != "all":
            states = [state for state in states if state.status == self._filter_mode]
        states.sort(key=self._sort_key)
        return states

    def _sort_key(self, state: FileProgressState):
        status_rank = {
            "running": 0,
            "queued": 1,
            "done": 2,
            "failed": 3,
            "cancelled": 4,
            "skipped": 5,
            "unsupported": 6,
        }
        if self._sort_mode == "status":
            return (status_rank.get(state.status, 9), state.index)
        if self._sort_mode == "name":
            return (state.name.lower(), state.index)
        if self._sort_mode == "elapsed":
            if state.started_at is None:
                elapsed = 0.0
            elif state.started_at > 1_000_000_000:
                elapsed = (state.finished_at or time.time()) - state.started_at
            else:
                elapsed = (state.finished_at or time.perf_counter()) - state.started_at
            return (-elapsed, state.index)
        if self._sort_mode == "unresolved":
            return (-state.unresolved, status_rank.get(state.status, 9), state.index)
        return (state.index,)

    def _cycle_filter(self) -> None:
        modes = ["all", "running", "queued", "done", "failed", "cancelled", "skipped", "unsupported"]
        current = modes.index(self._filter_mode) if self._filter_mode in modes else 0
        self._filter_mode = modes[(current + 1) % len(modes)]
        self._selected_row = 0
        self._scroll_offset = 0
        self._log_line(self._timestamped(f"control :: filter set to {self._filter_mode}"))

    def _cycle_sort(self) -> None:
        modes = ["index", "status", "name", "elapsed", "unresolved"]
        current = modes.index(self._sort_mode) if self._sort_mode in modes else 0
        self._sort_mode = modes[(current + 1) % len(modes)]
        self._selected_row = 0
        self._scroll_offset = 0
        self._log_line(self._timestamped(f"control :: sort set to {self._sort_mode}"))

    def _move_selection(self, delta: int) -> None:
        visible = self._visible_states()
        if not visible:
            self._selected_row = 0
            self._scroll_offset = 0
            return
        self._selected_row = max(0, min(len(visible) - 1, self._selected_row + delta))

    def _page_selection(self, direction: int) -> None:
        if self._screen is None:
            return
        height, _ = self._screen.getmaxyx()
        page = max(4, height - 13)
        self._move_selection(direction * page)

    def _jump_bottom(self) -> None:
        visible = self._visible_states()
        if not visible:
            self._selected_row = 0
            self._scroll_offset = 0
            return
        self._selected_row = len(visible) - 1

    def _trigger_selected(self) -> None:
        selected = self.selected_index()
        if selected is None:
            return
        if not self.prioritize_file(selected):
            state = self._states.get(selected)
            if state is not None:
                self._log_line(self._timestamped(f"control :: cannot trigger {state.name} from status {state.status}"))

    def _cancel_selected(self) -> None:
        selected = self.selected_index()
        if selected is None:
            return
        state = self._states.get(selected)
        if state is None:
            return
        changed = self.cancel_file(selected)
        if not changed and state.status not in {"queued", "running"}:
            self._log_line(self._timestamped(f"control :: cannot cancel {state.name} from status {state.status}"))

    def _clamp_viewport(self, total_visible: int, table_rows: int) -> None:
        if total_visible <= 0:
            self._selected_row = 0
            self._scroll_offset = 0
            return
        self._selected_row = max(0, min(total_visible - 1, self._selected_row))
        max_offset = max(0, total_visible - table_rows)
        if self._selected_row < self._scroll_offset:
            self._scroll_offset = self._selected_row
        elif self._selected_row >= self._scroll_offset + table_rows:
            self._scroll_offset = self._selected_row - table_rows + 1
        self._scroll_offset = max(0, min(max_offset, self._scroll_offset))

    def _log_line(self, line: str) -> None:
        entry = line if line.startswith("[") and len(line) > 9 and line[3] == ":" and line[6] == ":" else self._timestamped(line)
        if self._mode == "plain":
            print(entry, flush=True)
            return
        self._recent_logs.append(entry)
        if len(self._recent_logs) > 40:
            self._recent_logs = self._recent_logs[-40:]

    def _timestamped(self, line: str) -> str:
        return f"[{time.strftime('%H:%M:%S')}] {line}"

    def _prefix(self) -> str:
        snapshot = self.snapshot()
        finished = snapshot.completed + snapshot.skipped + snapshot.unsupported + snapshot.cancelled
        return (
            f"[done={finished}/{snapshot.total_files} active={snapshot.active} "
            f"queued={snapshot.queued} extracted={snapshot.extracted} unresolved={snapshot.unresolved}]"
        )

    def _state_elapsed(self, state: FileProgressState) -> str:
        if state.started_at is None:
            return "0.0s"
        if state.started_at > 1_000_000_000:
            end = state.finished_at if state.finished_at is not None else time.time()
        else:
            end = state.finished_at if state.finished_at is not None else time.perf_counter()
        return _format_elapsed(end - state.started_at)

    def _style_for_mode(self, mode: str) -> int:
        if "PAUSED" in mode:
            return self._style("warn")
        if "STOPPING" in mode or "ABORTING" in mode:
            return self._style("error")
        return self._style("success")

    def _style_for_state(self, state: FileProgressState) -> int:
        if state.status == "done":
            return self._style("success")
        if state.status == "running":
            return self._style("warn") if state.cancel_requested else self._style("accent")
        if state.status == "cancelled":
            return self._style("error")
        if state.status == "failed":
            return self._style("error")
        if state.status == "skipped":
            return self._style("dim")
        if state.status == "unsupported":
            return self._style("error")
        return self._style("plain")

    def _render_inspector(self, start_row: int, height: int, width: int) -> None:
        self._add_line(start_row, 0, "Inspector", width - 1, self._style("header"))
        selected = self.selected_index()
        if selected is None:
            self._add_line(start_row + 1, 0, "No row selected.", width - 1, self._style("plain"))
            return
        state = self._states.get(selected)
        if state is None:
            self._add_line(start_row + 1, 0, "Selected row not available.", width - 1, self._style("plain"))
            return
        lines = [
            f"Index: {state.index}",
            f"Name: {state.name}",
            f"Status: {state.status}",
            f"Lease status: {state.lease_status}",
            f"Lease owner: {state.lease_owner or '(unknown)'}",
            f"Stage: {state.stage}",
            f"Elapsed: {self._state_elapsed(state)}",
            f"Candidates: {state.candidates}  Accepted: {state.accepted}  Unresolved: {state.unresolved}",
            f"Priority boost: {state.priority_boost}  Cancel requested: {state.cancel_requested}",
            f"Detail: {state.detail or '(none)'}",
        ]
        for offset, line in enumerate(lines, start=1):
            target_row = start_row + offset
            if target_row >= height:
                break
            self._add_line(target_row, 0, line, width - 1, self._style("plain"))

    def _style(self, name: str) -> int:
        if not self._supports_color or self._curses is None:
            return 0
        curses = self._curses
        mapping = {
            "accent": curses.color_pair(1),
            "success": curses.color_pair(2),
            "warn": curses.color_pair(3),
            "error": curses.color_pair(4),
            "header": curses.color_pair(5) | curses.A_BOLD,
            "dim": curses.color_pair(7) | curses.A_DIM,
            "plain": curses.color_pair(7),
        }
        return mapping.get(name, 0)

    def _add_line(self, y: int, x: int, text: str, width: int, style: int = 0) -> None:
        if self._screen is None:
            return
        try:
            self._screen.addnstr(y, x, text, max(0, width), style)
        except Exception:
            return


def _format_elapsed(elapsed: float) -> str:
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    minutes, seconds = divmod(elapsed, 60)
    return f"{int(minutes)}m {seconds:.1f}s"

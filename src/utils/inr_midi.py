import bisect
import math
from collections import defaultdict

from miditoolkit import ControlChange, Instrument, MidiFile, Note, TempoChange


CONTINUOUS_KEYS = (
    "ioi",
    "duration",
    "velocity",
    "pedal_0",
    "pedal_25",
    "pedal_50",
    "pedal_75",
)


RAW_CONTINUOUS_KEYS = (
    "ioi_ms",
    "duration_ms",
    "velocity",
    "pedal_0",
    "pedal_25",
    "pedal_50",
    "pedal_75",
)

RAW_SHARED_KEYS = (
    "ioi_ms",
    "duration_ms",
    "velocity",
)

RAW_PEDAL4_KEYS = (
    "pedal_0",
    "pedal_25",
    "pedal_50",
    "pedal_75",
)

RAW_PEDAL2_KEYS = (
    "pedal_start",
    "pedal_ctrl",
)

RAW_PEDAL_START_VALLEY_KEYS = (
    "pedal_start",
    "has_valley",
)


def sorted_piano_notes(midi_obj):
    notes = []
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (note.start, note.pitch, note.end, note.velocity))


def sorted_pedal_controls(midi_obj):
    controls = []
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            controls.extend(cc for cc in instrument.control_changes if cc.number == 64)
    return sorted(controls, key=lambda cc: (cc.time, cc.value))


def normalize_time_ms(time_ms, max_time_ms=10000.0):
    clipped = min(max(float(time_ms), 0.0), float(max_time_ms))
    return math.log1p(clipped) / math.log1p(float(max_time_ms))


def denormalize_time_ms(time_norm, max_time_ms=10000.0):
    clipped = min(max(float(time_norm), 0.0), 1.0)
    return math.expm1(clipped * math.log1p(float(max_time_ms)))


def normalize_time_ms_for_inr_input(time_ms, normalization="legacy_log1p", max_time_ms=10000.0):
    normalization = str(normalization or "legacy_log1p").lower()
    if normalization in {"legacy_log1p", "log1p", "log1p_10000"}:
        return normalize_time_ms(time_ms, max_time_ms=max_time_ms)
    if normalization in {"scaled_log_5000_s10", "log1p_t_over_10_5000", "log1p_x_over_10_5000"}:
        clipped = min(max(float(time_ms), 0.0), 5000.0)
        return math.log1p(clipped / 10.0) / math.log1p(500.0)
    if normalization in {"log1p_t_over_50_5000", "log1p_x_over_50_5000"}:
        clipped = min(max(float(time_ms), 0.0), 5000.0)
        return math.log1p(clipped / 50.0) / math.log1p(100.0)
    if normalization in {"log1p_t_over_100_5000", "log1p_x_over_100_5000"}:
        clipped = min(max(float(time_ms), 0.0), 5000.0)
        return math.log1p(clipped / 100.0) / math.log1p(50.0)
    if normalization in {"linear_5000", "raw_linear_5000"}:
        return min(max(float(time_ms), 0.0), 5000.0) / 5000.0
    raise ValueError(f"Unsupported timing normalization: {normalization}")


def denormalize_time_ms_from_inr_input(time_norm, normalization="legacy_log1p", max_time_ms=10000.0):
    normalization = str(normalization or "legacy_log1p").lower()
    clipped = min(max(float(time_norm), 0.0), 1.0)
    if normalization in {"legacy_log1p", "log1p", "log1p_10000"}:
        return denormalize_time_ms(clipped, max_time_ms=max_time_ms)
    if normalization in {"scaled_log_5000_s10", "log1p_t_over_10_5000", "log1p_x_over_10_5000"}:
        return math.expm1(clipped * math.log1p(500.0)) * 10.0
    if normalization in {"log1p_t_over_50_5000", "log1p_x_over_50_5000"}:
        return math.expm1(clipped * math.log1p(100.0)) * 50.0
    if normalization in {"log1p_t_over_100_5000", "log1p_x_over_100_5000"}:
        return math.expm1(clipped * math.log1p(50.0)) * 100.0
    if normalization in {"linear_5000", "raw_linear_5000"}:
        return clipped * 5000.0
    raise ValueError(f"Unsupported timing normalization: {normalization}")


def old_continuous_row_to_raw(row, max_time_ms=10000.0):
    return [
        denormalize_time_ms(row[0], max_time_ms=max_time_ms),
        denormalize_time_ms(row[1], max_time_ms=max_time_ms),
        min(max(float(row[2]), 0.0), 1.0) * 127.0,
        *[min(max(float(value), 0.0), 1.0) * 127.0 for value in row[3:7]],
    ]


def raw_row_to_model_continuous(row, timing_normalization="legacy_log1p", max_time_ms=10000.0):
    return [
        normalize_time_ms_for_inr_input(
            row[0],
            normalization=timing_normalization,
            max_time_ms=max_time_ms,
        ),
        normalize_time_ms_for_inr_input(
            row[1],
            normalization=timing_normalization,
            max_time_ms=max_time_ms,
        ),
        min(max(float(row[2]), 0.0), 127.0) / 127.0,
        *[min(max(float(value), 0.0), 127.0) / 127.0 for value in row[3:7]],
    ]


def raw_rows_to_model_continuous(rows, timing_normalization="legacy_log1p", max_time_ms=10000.0):
    return [
        raw_row_to_model_continuous(
            row,
            timing_normalization=timing_normalization,
            max_time_ms=max_time_ms,
        )
        for row in rows
    ]


def old_continuous_rows_to_raw(rows, max_time_ms=10000.0):
    return [old_continuous_row_to_raw(row, max_time_ms=max_time_ms) for row in rows]


def raw_row_to_epr_bins(row, timing_bins=5000, value_bins=128):
    timing_max = int(timing_bins) - 1
    value_max = int(value_bins) - 1
    return [
        min(max(int(round(float(row[0]))), 0), timing_max),
        min(max(int(round(float(row[1]))), 0), timing_max),
        min(max(int(round(float(row[2]))), 0), value_max),
        *[min(max(int(round(float(value))), 0), value_max) for value in row[3:7]],
    ]


def raw_rows_to_epr_bins(rows, timing_bins=5000, value_bins=128):
    return [raw_row_to_epr_bins(row, timing_bins=timing_bins, value_bins=value_bins) for row in rows]


def _tick_to_ms_mapping(midi_obj):
    tick_to_time = midi_obj.get_tick_to_time_mapping()
    return [time_sec * 1000.0 for time_sec in tick_to_time]


def _time_at_tick_ms(tick_to_ms, tick):
    if tick < len(tick_to_ms):
        return tick_to_ms[tick]
    if not tick_to_ms:
        return 0.0
    return tick_to_ms[-1]


def _cc_value_at_ms(cc_times_ms, cc_values, query_ms):
    idx = bisect.bisect_right(cc_times_ms, query_ms)
    if idx == 0:
        return 0
    return cc_values[idx - 1]


def _cc_extreme_value_between_ms(cc_times_ms, cc_values, start_ms, end_ms, start_value, next_start_value):
    if end_ms <= start_ms:
        return start_value
    lower = min(start_value, next_start_value)
    upper = max(start_value, next_start_value)
    left = bisect.bisect_right(cc_times_ms, start_ms)
    right = bisect.bisect_right(cc_times_ms, end_ms)
    values = [next_start_value]
    values.extend(cc_values[left:right])

    best_value = _cc_value_at_ms(cc_times_ms, cc_values, start_ms + (end_ms - start_ms) * 0.5)
    best_distance = -1.0
    for value in values:
        distance = max(lower - value, 0.0, value - upper)
        if distance > best_distance:
            best_distance = distance
            best_value = value
    if best_distance <= 0.0:
        return _cc_value_at_ms(cc_times_ms, cc_values, start_ms + (end_ms - start_ms) * 0.5)
    return best_value


def _cc_min_value_between_ms(cc_times_ms, cc_values, start_ms, end_ms, start_value, next_start_value):
    if end_ms <= start_ms:
        return min(start_value, next_start_value)
    left = bisect.bisect_right(cc_times_ms, start_ms)
    right = bisect.bisect_left(cc_times_ms, end_ms)
    values = [start_value, next_start_value]
    values.extend(cc_values[left:right])
    return min(values)


def _deduplicate_controls(control_changes):
    output = []
    last_value = None
    for cc in sorted(control_changes, key=lambda item: (item.time, item.value)):
        if cc.value != last_value:
            output.append(cc)
            last_value = cc.value
    return output


def midi_to_note_features(
    midi_obj,
    notes=None,
    max_time_ms=10000.0,
    normalize=True,
    force_monotonic_starts=False,
    include_pedal2=False,
    include_pedal_start_valley=False,
    pedal_binary_threshold=64.0,
):
    """Convert a MIDI object to note-level pitch and continuous features.

    Time features are computed in real milliseconds using the MIDI tempo map and
    kept as floats before optional log normalization.
    """
    if notes is None:
        notes = sorted_piano_notes(midi_obj)
    else:
        notes = list(notes)

    tick_to_ms = _tick_to_ms_mapping(midi_obj)
    pedal_controls = sorted_pedal_controls(midi_obj)
    pedal_times_ms = [_time_at_tick_ms(tick_to_ms, cc.time) for cc in pedal_controls]
    pedal_values = [cc.value for cc in pedal_controls]

    raw_starts_ms = [_time_at_tick_ms(tick_to_ms, note.start) for note in notes]
    raw_ends_ms = [_time_at_tick_ms(tick_to_ms, note.end) for note in notes]
    starts_ms = sorted(raw_starts_ms) if force_monotonic_starts else raw_starts_ms

    pitches = []
    continuous = []
    pedal2 = []
    pedal_start_valley = []
    last_start_ms = 0.0

    for idx, note in enumerate(notes):
        start_ms = starts_ms[idx]
        next_start_ms = starts_ms[idx + 1] if idx + 1 < len(starts_ms) else start_ms + 4990.0
        next_ioi_ms = max(next_start_ms - start_ms, 0.0)

        ioi_ms = max(start_ms - last_start_ms, 0.0)
        duration_ms = max(raw_ends_ms[idx] - raw_starts_ms[idx], 0.0)
        last_start_ms = start_ms

        pedal_samples = [
            _cc_value_at_ms(pedal_times_ms, pedal_values, start_ms),
            _cc_value_at_ms(pedal_times_ms, pedal_values, start_ms + next_ioi_ms * 0.25),
            _cc_value_at_ms(pedal_times_ms, pedal_values, start_ms + next_ioi_ms * 0.50),
            _cc_value_at_ms(pedal_times_ms, pedal_values, start_ms + next_ioi_ms * 0.75),
        ]
        pedal_start = pedal_samples[0]
        next_pedal_start = _cc_value_at_ms(pedal_times_ms, pedal_values, next_start_ms)
        pedal_ctrl = _cc_extreme_value_between_ms(
            pedal_times_ms,
            pedal_values,
            start_ms,
            next_start_ms,
            pedal_start,
            next_pedal_start,
        )
        pedal_min = _cc_min_value_between_ms(
            pedal_times_ms,
            pedal_values,
            start_ms,
            next_start_ms,
            pedal_start,
            next_pedal_start,
        )
        threshold = float(pedal_binary_threshold)
        has_valley = (
            float(pedal_start) >= threshold
            and float(next_pedal_start) >= threshold
            and float(pedal_min) < threshold
        )

        if normalize:
            ioi_value = normalize_time_ms(ioi_ms, max_time_ms=max_time_ms)
            duration_value = normalize_time_ms(duration_ms, max_time_ms=max_time_ms)
            velocity_value = min(max(float(note.velocity) / 127.0, 0.0), 1.0)
            pedal_values_out = [min(max(float(value) / 127.0, 0.0), 1.0) for value in pedal_samples]
            pedal2_values_out = [
                min(max(float(pedal_start) / 127.0, 0.0), 1.0),
                min(max(float(pedal_ctrl) / 127.0, 0.0), 1.0),
            ]
            pedal_start_valley_values_out = [
                min(max(float(pedal_start) / 127.0, 0.0), 1.0),
                1.0 if has_valley else 0.0,
            ]
        else:
            ioi_value = ioi_ms
            duration_value = duration_ms
            velocity_value = min(max(float(note.velocity), 0.0), 127.0)
            pedal_values_out = [min(max(float(value), 0.0), 127.0) for value in pedal_samples]
            pedal2_values_out = [
                min(max(float(pedal_start), 0.0), 127.0),
                min(max(float(pedal_ctrl), 0.0), 127.0),
            ]
            pedal_start_valley_values_out = [
                min(max(float(pedal_start), 0.0), 127.0),
                1.0 if has_valley else 0.0,
            ]

        pitches.append(int(note.pitch))
        continuous.append(
            [
                ioi_value,
                duration_value,
                velocity_value,
                *pedal_values_out,
            ]
        )
        pedal2.append(pedal2_values_out)
        pedal_start_valley.append(pedal_start_valley_values_out)

    result = {
        "pitch": pitches,
        "continuous": continuous,
        "shared": [row[:3] for row in continuous],
        "pedal4": [row[3:7] for row in continuous],
    }
    if include_pedal2:
        result["pedal2"] = pedal2
    if include_pedal_start_valley:
        result["pedal_start_valley"] = pedal_start_valley
    return result


def note_features_to_midi(
    pitch,
    continuous,
    target_ticks_per_beat=500,
    target_tempo=120,
    max_time_ms=10000.0,
    normalized=True,
    pedal_start_valley=None,
    valley_phase=0.5,
    valley_restore_phase=0.9,
):
    """Build a MIDI object from pitch and predicted continuous features."""
    def _maybe_scale_midi_value(value):
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            numeric *= 127.0
        return min(max(numeric, 0.0), 127.0)

    notes = []
    control_changes = []

    current_ms = 0.0
    ioi_values = []
    duration_values = []
    velocity_values = []
    pedal_values = []
    valley_values = []

    for idx, row in enumerate(continuous):
        if normalized:
            ioi_ms = denormalize_time_ms_from_inr_input(
                row[0],
                normalization=normalized if isinstance(normalized, str) else "legacy_log1p",
                max_time_ms=max_time_ms,
            )
            duration_ms = denormalize_time_ms_from_inr_input(
                row[1],
                normalization=normalized if isinstance(normalized, str) else "legacy_log1p",
                max_time_ms=max_time_ms,
            )
            velocity = int(round(min(max(float(row[2]), 0.0), 1.0) * 127.0))
            pedals = [int(round(min(max(float(value), 0.0), 1.0) * 127.0)) for value in row[3:7]]
        else:
            ioi_ms = max(float(row[0]), 0.0)
            duration_ms = max(float(row[1]), 0.0)
            velocity = int(round(_maybe_scale_midi_value(row[2])))
            pedals = [int(round(_maybe_scale_midi_value(value))) for value in row[3:7]]
        ioi_values.append(ioi_ms)
        duration_values.append(duration_ms)
        velocity_values.append(velocity)
        pedal_values.append(pedals)
        if pedal_start_valley is not None:
            valley_row = pedal_start_valley[idx]
            valley_values.append(1.0 if float(valley_row[1]) >= 0.5 else 0.0)
        else:
            valley_values.append(0.0)

    for idx, pitch_value in enumerate(pitch):
        current_ms += ioi_values[idx]
        start_tick = int(round(current_ms))
        end_tick = max(start_tick + 1, int(round(current_ms + duration_values[idx])))
        note_velocity = max(1, velocity_values[idx])
        notes.append(Note(note_velocity, int(pitch_value), start_tick, end_tick))

        next_ioi = ioi_values[idx + 1] if idx + 1 < len(ioi_values) else 4990.0
        sample_times = [
            current_ms,
            current_ms + next_ioi * 0.25,
            current_ms + next_ioi * 0.50,
            current_ms + next_ioi * 0.75,
        ]
        for value, sample_time in zip(pedal_values[idx], sample_times):
            control_changes.append(ControlChange(64, value, int(round(sample_time))))
        if valley_values[idx] >= 0.5 and idx + 1 < len(pedal_values):
            valley_time = current_ms + next_ioi * min(max(float(valley_phase), 0.0), 1.0)
            restore_time = current_ms + next_ioi * min(max(float(valley_restore_phase), 0.0), 1.0)
            restore_value = int(round(_maybe_scale_midi_value(pedal_values[idx + 1][0])))
            control_changes.append(ControlChange(64, 0, int(round(valley_time))))
            control_changes.append(ControlChange(64, restore_value, int(round(restore_time))))

    control_changes = _deduplicate_controls(control_changes)

    max_tick = 0
    if notes:
        max_tick = max(max_tick, max(note.end for note in notes))
    if control_changes:
        max_tick = max(max_tick, max(cc.time for cc in control_changes))

    output = MidiFile(ticks_per_beat=target_ticks_per_beat)
    output.tempo_changes.append(TempoChange(target_tempo, 0))
    output.instruments.append(
        Instrument(
            program=0,
            is_drum=False,
            name="Piano",
            notes=notes,
            control_changes=control_changes,
        )
    )
    output.max_tick = max_tick + 1
    return output

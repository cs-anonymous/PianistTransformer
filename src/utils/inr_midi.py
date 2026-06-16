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

        if normalize:
            ioi_value = normalize_time_ms(ioi_ms, max_time_ms=max_time_ms)
            duration_value = normalize_time_ms(duration_ms, max_time_ms=max_time_ms)
        else:
            ioi_value = ioi_ms
            duration_value = duration_ms

        pitches.append(int(note.pitch))
        continuous.append(
            [
                ioi_value,
                duration_value,
                min(max(float(note.velocity) / 127.0, 0.0), 1.0),
                *[min(max(float(value) / 127.0, 0.0), 1.0) for value in pedal_samples],
            ]
        )

    return {
        "pitch": pitches,
        "continuous": continuous,
    }


def note_features_to_midi(
    pitch,
    continuous,
    target_ticks_per_beat=500,
    target_tempo=120,
    max_time_ms=10000.0,
    normalized=True,
):
    """Build a MIDI object from pitch and predicted continuous features."""
    notes = []
    control_changes = []

    current_ms = 0.0
    ioi_values = []
    duration_values = []
    velocity_values = []
    pedal_values = []

    for row in continuous:
        if normalized:
            ioi_ms = denormalize_time_ms(row[0], max_time_ms=max_time_ms)
            duration_ms = denormalize_time_ms(row[1], max_time_ms=max_time_ms)
        else:
            ioi_ms = max(float(row[0]), 0.0)
            duration_ms = max(float(row[1]), 0.0)
        ioi_values.append(ioi_ms)
        duration_values.append(duration_ms)
        velocity_values.append(int(round(min(max(float(row[2]), 0.0), 1.0) * 127.0)))
        pedal_values.append([int(round(min(max(float(value), 0.0), 1.0) * 127.0)) for value in row[3:7]])

    for idx, pitch_value in enumerate(pitch):
        current_ms += ioi_values[idx]
        start_tick = int(round(current_ms))
        end_tick = max(start_tick + 1, int(round(current_ms + duration_values[idx])))
        notes.append(Note(velocity_values[idx], int(pitch_value), start_tick, end_tick))

        next_ioi = ioi_values[idx + 1] if idx + 1 < len(ioi_values) else 4990.0
        sample_times = [
            current_ms,
            current_ms + next_ioi * 0.25,
            current_ms + next_ioi * 0.50,
            current_ms + next_ioi * 0.75,
        ]
        for value, sample_time in zip(pedal_values[idx], sample_times):
            control_changes.append(ControlChange(64, value, int(round(sample_time))))

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

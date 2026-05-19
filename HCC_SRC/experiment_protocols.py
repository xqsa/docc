"""Named experiment presets for HCC runs."""


def _normalize_record_fes(max_fes, record_fes):
    normalized = []
    for point in record_fes:
        point = int(point)
        if point <= 0:
            raise ValueError("record_fes must contain positive evaluation counts")
        if point > max_fes:
            raise ValueError("record_fes cannot exceed max_fes")
        if normalized and point <= normalized[-1]:
            raise ValueError("record_fes must be strictly increasing")
        normalized.append(point)
    return normalized


def _scaled_record_fes(max_fes, fractions):
    record_fes = []
    for fraction in fractions:
        point = max(1, min(max_fes, int(round(max_fes * fraction))))
        if not record_fes or point > record_fes[-1]:
            record_fes.append(point)
    if record_fes[-1] != max_fes:
        record_fes.append(max_fes)
    return record_fes


def _build_protocol(name, max_fes, cycle_num, record_fes):
    return {
        "name": name,
        "max_fes": int(max_fes),
        "cycle_num": int(cycle_num),
        "record_fes": _normalize_record_fes(int(max_fes), record_fes),
    }


_PROTOCOLS = {
    "smoke": _build_protocol(
        "smoke",
        max_fes=2_000,
        cycle_num=5,
        record_fes=_scaled_record_fes(2_000, [0.10, 0.25, 0.50, 0.75, 1.0]),
    ),
    "paper": _build_protocol(
        "paper",
        max_fes=3_000_000,
        cycle_num=25,
        record_fes=[120_000, 200_000, 1_000_000, 2_000_000, 3_000_000],
    ),
}


def protocol_choices():
    return list(_PROTOCOLS.keys())


def resolve_protocol(name):
    key = name.lower()
    if key not in _PROTOCOLS:
        raise ValueError(f"Unknown protocol: {name}")
    protocol = _PROTOCOLS[key]
    return {
        "name": protocol["name"],
        "max_fes": protocol["max_fes"],
        "cycle_num": protocol["cycle_num"],
        "record_fes": list(protocol["record_fes"]),
    }

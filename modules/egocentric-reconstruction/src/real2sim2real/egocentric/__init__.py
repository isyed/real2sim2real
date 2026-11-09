"""Independent egocentric reconstruction; heavy backends load only in workers."""


def descriptor():
    return (
        "real2sim2real.egocentric.stage",
        {"calibrated-capture"},
        "object-reconstruction",
    )
